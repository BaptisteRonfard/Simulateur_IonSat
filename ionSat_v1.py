import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# import simulation related support
from Basilisk.simulation import spacecraft
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
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

def is_visible(stations_list, r_sc):
    """
    Evaluates if a satellite is visible from com stations, angle_lim is the angle of visibility
    """
    angle_lim = 15 * np.pi / 180

    for r_stat in stations_list:
        if np.dot(r_stat, r_sc - r_stat) / (np.linalg.norm(r_stat) * np.linalg.norm(r_sc - r_stat)) < np.cos(angle_lim):
            return True
    return False

def run(thrust, initialAlt=250, deorbitAlt=200, model="exponential"):
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
    oe = orbitalMotion.ClassicElements()
    oe.a = planet.radEquator + initialAlt * 1000  # meters
    oe.e = 0.0001
    oe.i = 33.3 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    oe = orbitalMotion.rv2elem(mu, rN, vN)
    # this stores consistent initial orbit elements; with circular or equatorial orbit, some angles are arbitrary

    # To set the spacecraft initial conditions, the following initial position and velocity variables are set:
    scObject.hub.r_CN_NInit = rN  # m   - r_BN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_BN_N

    #initialize micro-thruster

    microThruster = extForceTorque.ExtForceTorque()
    microThruster.ModelTag = "MicroThrust"
    microThrusterTaskName = "thrust"
    microThruster.extForce_B = vN * thrust
    dynProcess.addTask(scSim.CreateNewTask(microThrusterTaskName, simulationTimeStep))
    scSim.AddModelToTask(microThrusterTaskName, microThruster)

    # Link spacecraft to thrust
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

    # Setup data logging before the simulation is initialized
    dataRec = scObject.scStateOutMsg.recorder(samplingTime)
    dataAtmoLog = atmo.envOutMsgs[0].recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, dataRec)
    scSim.AddModelToTask(simTaskName, dataAtmoLog)
    scSim.AddVariableForLogging('DragEff.forceExternal_B', samplingTime, StartIndex=0, StopIndex=2)

    # Vizard Visualization Option
    # ---------------------------
    # If you wish to transmit the simulation data to the United based Vizard Visualization application,
    # then uncomment the following
    # line from the python scenario script.  This will cause the BSK simulation data to
    # be stored in a binary file inside the _VizFiles sub-folder with the scenario folder.  This file can be read in by
    # Vizard and played back after running the BSK simulation.
    # To enable this, uncomment this line:]
    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject,
                                              # saveFile=__file__
                                              # liveStream=True
                                              )

    # initialize Simulation
    scSim.InitializeSimulation()

    # Repeatedly step the simulation ahead and break once deorbit altitude encountered
    steps = 0
    exit_sim = False
    r = 0
    while not exit_sim:
        steps += 1
        scSim.ConfigureStopTime(steps * simulationTime)
        scSim.ExecuteSimulation()
        r = orbitalMotion.rv2elem(mu, dataRec.r_BN_N[-1], dataRec.v_BN_N[-1]).rmag
        alt = (r - planet.radEquator) / 1000  # km
        visible = is_visible(station_list, dataRec.r_BN_N[-1])
        if alt < deorbitAlt or alt > 1e10:
            exit_sim = True
        if visible:
            exit_sim = True

    OutMessage = pd.Dataframe({'position':np.array([dataRec.r_BN_N[-1]]), 'velocity':np.array([dataRec.v_BN_N[-1]]), 'altitude':np.array([r])})
    OutMessage.to_csv(dir+'out_message.csv') # saves final data at the end of the sim in a csv located in current directory

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Basic Simulator for IonSat')

    parser.add_argument('--m',nargs='?', type=str,help = 'Type Y if you want to modify thrust, type nothing otherwise')
    parser.add_argument('--thrust',nargs='?', type=float,help = 'Give new thrust value in mN')

    args = parser.parse_args()

    if args.m == 'Y':
        if args.thrust:
            if os.path.exists(dir+'out_message.csv') == True:
                run(args.thrust * 0.001, initialAlt=250, deorbitAlt=200, model="exponential")
            else:
                run
    
    run(t, initialAlt=250, deorbitAlt=200, model="exponential")