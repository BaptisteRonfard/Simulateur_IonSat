import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
# import simulation related support
from Basilisk.simulation import spacecraft, groundLocation
from Basilisk.utilities import (SimulationBaseClass, astroFunctions, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport, vizSupport, simSetPlanetEnvironment)

# import atmosphere and drag modules
from Basilisk.simulation import exponentialAtmosphere, msisAtmosphere, dragDynamicEffector, extForceTorque

# always import the Basilisk messaging support
from Basilisk.architecture import messaging

# The path to the location of Basilisk, used to get the location of supporting data
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

t = 0
dir = './'

R = 6400000
stations_list = np.array([[0.5*R, 0.3*R, 0.4*R]])

def is_visible(stations_list, r_sc):
    """
    Evaluates if a satellite is visible from com stations, angle_lim is the angle of visibility
    """
    angle_lim = 15 * np.pi / 180

    for r_stat in stations_list:
        if np.dot(r_stat, (r_sc - r_stat)) / (np.linalg.norm(r_stat) * np.linalg.norm(r_sc - r_stat)) < np.cos(angle_lim):
            return 1
    return 0

def bool_to_bin(statement):

    if statement:
        return 1
    return 0

def run(thrust, initialAlt=(412 + 419)/2, deorbitAlt=190, model="exponential"):
    """
    Initialize a satellite with drag and propagate until it falls below a deorbit altitude. Note that an excessively
    low deorbit_alt can lead to intersection with the Earth prior to deorbit being detected, causing some terms to blow
    up and the simulation to terminate.

    Args:
        show_plots (bool): Toggle plotting on/off
        initialAlt (float): Starting altitude in km
        deorbitAlt (float): Terminal altitude in km
        model (str): ["exponential", "msis"]

    Returns:
        Dictionary of figure handles
    """
    # Create simulation and dynamics process
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(15.)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Initialize atmosphere model and add to sim
    if model == "exponential":
        atmo = exponentialAtmosphere.ExponentialAtmosphere()
        atmo.ModelTag = "ExpAtmo"
        simSetPlanetEnvironment.exponentialAtmosphere(atmo, "earth")
    elif model == "msis":
        atmo = msisAtmosphere.MsisAtmosphere()
        atmo.ModelTag = "MsisAtmo"

        ap = 8
        f107 = 110
        sw_msg = {
            "ap_24_0": ap, "ap_3_0": ap, "ap_3_-3": ap, "ap_3_-6": ap, "ap_3_-9": ap,
            "ap_3_-12": ap, "ap_3_-15": ap, "ap_3_-18": ap, "ap_3_-21": ap, "ap_3_-24": ap,
            "ap_3_-27": ap, "ap_3_-30": ap, "ap_3_-33": ap, "ap_3_-36": ap, "ap_3_-39": ap,
            "ap_3_-42": ap, "ap_3_-45": ap, "ap_3_-48": ap, "ap_3_-51": ap, "ap_3_-54": ap,
            "ap_3_-57": ap, "f107_1944_0": f107, "f107_24_-24": f107
        }

        swMsgList = []
        for c, val in enumerate(sw_msg.values()):
            swMsgData = messaging.SwDataMsgPayload()
            swMsgData.dataValue = val
            swMsgList.append(messaging.SwDataMsg().write(swMsgData))
            atmo.swDataInMsgs[c].subscribeTo(swMsgList[-1])
    else:
        raise ValueError(f"{model} not a valid model!")

    scSim.AddModelToTask(simTaskName, atmo)

    # Initialize drag effector and add to sim
    projArea = 10.0  # drag area in m^2
    dragCoeff = 2.2  # drag coefficient
    dragEffector = dragDynamicEffector.DragDynamicEffector()
    dragEffector.ModelTag = "DragEff"
    dragEffectorTaskName = "drag"
    dragEffector.coreParams.projectedArea = projArea
    dragEffector.coreParams.dragCoeff = dragCoeff
    dynProcess.addTask(scSim.CreateNewTask(dragEffectorTaskName, simulationTimeStep))
    scSim.AddModelToTask(dragEffectorTaskName, dragEffector)

    # Set up the spacecraft
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    scSim.AddModelToTask(simTaskName, scObject)

    # Link spacecraft to drag model
    atmo.addSpacecraftToModel(scObject.scStateOutMsg)
    scObject.addDynamicEffector(dragEffector)
    # and drag model to atmosphere model
    dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])

    # Set up gravity
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createEarth()
    mu = planet.mu
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

    # Set up a circular orbit using classical orbit elements
    if os.path.exists(dir + 'out_message.csv') == False:
        
        E = orbitalMotion.M2E(234.8857 * macros.D2R, 0.0004463)
        f = orbitalMotion.E2f(E, 0.0004463)

        oe = orbitalMotion.ClassicElements()
        oe.a = planet.radEquator + initialAlt * 1000 # meters
        oe.e = 	0.0004463
        oe.i = 	51.6453 * macros.D2R
        oe.Omega = 350.6526 * macros.D2R
        oe.omega = 281.1231 * macros.D2R
        oe.f = f
        rN, vN = orbitalMotion.elem2rv(mu, oe)
        scObject.hub.r_CN_NInit = rN  
        scObject.hub.v_CN_NInit = vN

    else:

        orbit_elem = pd.read_csv(dir+'out_message.csv')
        rN, vN = orbit_elem['position'], orbit_elem['velocity']
        oe = orbitalMotion.rv2elem(mu, rN, vN)
        scObject.hub.r_CN_NInit = rN  # m - r_CN_N
        scObject.hub.v_CN_NInit = vN

    #initialize micro-thruster

    microThruster = extForceTorque.ExtForceTorque()
    microThruster.ModelTag = "MicroThrust"
    microThrusterTaskName = "thrust"
    microThruster.extForce_B = vN * thrust / np.linalg.norm(vN)
    dynProcess.addTask(scSim.CreateNewTask(microThrusterTaskName, simulationTimeStep))
    scSim.AddModelToTask(microThrusterTaskName, microThruster)
    scObject.addDynamicEffector(microThruster)

    # set the simulation time increments
    n = np.sqrt(mu / oe.a / oe.a / oe.a)
    P = 2. * np.pi / n

    # fraction of initial orbit period to step the simulation by
    if model == "exponential":
        orbit_frac = 0.1
    elif model == "msis":
        orbit_frac = 0.03  # msis deorbits more quickly
    simulationTime = macros.sec2nano(orbit_frac * P)
    numDataPoints = int(10000 * orbit_frac)  # per orbit_fraction at initial orbit conditions
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

#set up the observation station in Toulouse

    groundStation = groundLocation.GroundLocation()
    groundStation.ModelTag = "ToulouseGroundStation"
    groundStation.planetRadius = astroFunctions.E_radius*1e3  # meters
    groundStation.specifyLocation(np.radians(43.587101), np.radians(1.493185), 140)
    groundStation.minimumElevation = np.radians(1.)
    groundStation.maximumRange = 10000000  # meters
    groundStation.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, groundStation)

    # Setup data logging before the simulation is initialized
    dataRec = scObject.scStateOutMsg.recorder(samplingTime)
    dataRecAccess = groundStation.accessOutMsgs[-1].recorder()
    dataAtmoLog = atmo.envOutMsgs[0].recorder(samplingTime)

    scSim.AddModelToTask(simTaskName, dataRec)
    scSim.AddModelToTask(simTaskName, dataAtmoLog)
    scSim.AddModelToTask(simTaskName, dataRecAccess)
    scSim.AddVariableForLogging('DragEff.forceExternal_B', samplingTime, StartIndex=0, StopIndex=2)

    # initialize Simulation
    scSim.InitializeSimulation()
    posRef = scObject.dynManager.getStateObject("hubPosition")
    velRef = scObject.dynManager.getStateObject("hubVelocity")

    # Repeatedly step the simulation ahead and break once deorbit altitude encountered
    steps = 0
    exit_sim = False
    visible = 0
    r = 0
    rVt = 0
    vVt = 0
    while exit_sim < 2:

        steps += 1
        scSim.ConfigureStopTime(steps * simulationTime)
        scSim.ExecuteSimulation()

        rVt = unitTestSupport.EigenVector3d2np(posRef.getState())
        vVt = unitTestSupport.EigenVector3d2np(velRef.getState())
        r = orbitalMotion.rv2elem(mu, dataRec.r_BN_N[-1], dataRec.v_BN_N[-1]).rmag
        alt = (r - planet.radEquator) / 1000  # km

        print(str(datetime.timedelta(seconds=steps*orbit_frac*P)))

        if alt < deorbitAlt:
            exit_sim += 2
            print("the satellite escaped the earth's gravitational attraction")

        if alt > 1e10:
            exit_sim += 2
            print("the satellite was deorbited by the drag")

        if visible != dataRecAccess.hasAccess[-1]:

            print("test "+str(dataRecAccess.hasAccess[-1]))
            visible += 1
            exit_sim += 1

#sim runs while the the sat is invisible and then enters the visibility zone, and stops when it exits the visibility zone
    
    if os.path.exists(dir+'out_message.csv') == False:

        OutMessage = pd.DataFrame({'position':rVt, 'velocity':vVt}) 
        #'altitude':np.array([r]).tolist()})
        OutMessage.to_csv(dir+'out_message.csv') # saves final data at the end of the sim in a csv located in current directory

    else:

        os.remove(dir+'out_message.csv')
        OutMessage = pd.DataFrame({'position':rVt, 'velocity':vVt})
        OutMessage.to_csv(dir+'out_message.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Basic Simulator for IonSat')

    parser.add_argument('--m',nargs='?', type=str,help = 'Type Y if you want to modify thrust, type nothing otherwise')
    parser.add_argument('--thrust',nargs='?', type=float,help = 'Give new thrust value in mN')

    args = parser.parse_args()

    if args.m == 'Y':
        if args.thrust:
            run(args.thrust, initialAlt=(412 + 419)/2, deorbitAlt=190, model="exponential")
        else:
            raise ValueError(f'Please specify a thrust value')

    else:
        run(0, initialAlt=(412 + 419)/2, deorbitAlt=190, model="exponential")