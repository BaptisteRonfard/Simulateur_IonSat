#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

r"""
Overview
--------

Demonstrates how to stabilize the attitude tumble without translational motion.
This script sets up a 6-DOF spacecraft, but without specifying any orbital motion.  Thus,
this scenario simulates the spacecraft translating in deep space.  The scenario is a
version of :ref:`scenarioAttitudePointing` where the :ref:`mrpPD` feedback control
module is replaced with an equivalent python based BSK MRP PD control module.

The script is found in the folder ``basilisk/examples`` and executed by using::

    python3 scenarioAttitudePointingPY.py

As with :ref:`scenarioAttitudePointing`, when
the simulation completes 3 plots are shown for the MRP attitude history, the rate
tracking errors, as well as the control torque vector.

The MRP PD control module in this script is a class called ``PythonMRPPD``.  Note that it has the
same setup and update routines as are found with a C/C++ Basilisk module.

To use a Python module in a simulation script, not that the python modules must be added to a special python specific
process and task list.  This is done with the commands::

    pyModulesProcess = scSim.CreateNewPythonProcess(pyProcessName, 9)
    pyModulesProcess.createPythonTask(pyTaskName, simulationTimeStep, True, -1)

Note that the python processes are always evaluated after the regular C/C++ processes.  Thus, the priority number
only controls the order of the python processes, not the python process execution relative to regular
Basilisk processes.

Creating an instance of the Python module is done with the code::

    pyMRPPD = PythonMRPPD("pyMRP_PD", True, 100)
    pyMRPPD.K = 3.5
    pyMRPPD.P = 30.0
    pyModulesProcess.addModelToTask(pyTaskName, pyMRPPD)

The first argument is the module tag string, the second is a bool argument specifying if the module is active
or note, and the last is the priority value for this module.  The next step is to configure the module
variables as you do with any other Basilisk module.  Finally, the module is added to the special task
list specifically for executing python modules.



Illustration of Simulation Results
----------------------------------

::

    show_plots = True

Here a small initial tumble is simulated.  The
resulting attitude and control torque histories are shown below.  The spacecraft quickly
regains a stable orientation without tumbling past 180 degrees.

.. image:: /_images/Scenarios/scenarioAttitudePointingPy1.svg
   :align: center

.. image:: /_images/Scenarios/scenarioAttitudePointingPy2.svg
   :align: center

"""

#
# Basilisk Scenario Script and Integrated Test
#
# Purpose:  Integrated test showing how to setup and run a Python BSK module with C/C++ modules
# Author:   Hanspeter Schaub
# Creation Date:  Jan. 16, 2021
#
print("début du code")
import os
import numpy as np
import math as m
from Basilisk.utilities import RigidBodyKinematics
from Basilisk.utilities import orbitalMotion as om
from Basilisk.simulation import planetEphemeris

# import RigidBodyKinematics
from Basilisk.utilities import RigidBodyKinematics

# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
import matplotlib.pyplot as plt
from Basilisk.utilities import macros
from Basilisk.utilities import simulationArchTypes

# import simulation related support
from Basilisk.simulation import spacecraft
from Basilisk.utilities import simIncludeGravBody
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav

# import FSW Algorithm related support
# from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.fswAlgorithms import attTrackingError

# import message declarations
from Basilisk.architecture import messaging

# attempt to import vizard
from Basilisk.utilities import vizSupport

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

#For me : RigidBodyKinematics and planetEphemeris
from Basilisk.utilities import RigidBodyKinematics
from Basilisk.utilities import orbitalMotion as om



""" __________________________________________________For my simulation________________________________________________________________"""

#Constants
timestep = macros.sec2nano(.1)

#Functions
def MRP_diff(mrp1, mrp2):
    res = np.zeros(3)
    norm1 = np.linalg.norm(mrp1)
    norm2 = np.linalg.norm(mrp2)
    res = ((1-norm2**2)*mrp1 - (1-norm1**2)*mrp2 + 2*np.cross(mrp1,mrp2))/(1+norm1**2 * norm2**2 + 2*np.dot(mrp1,mrp2))
    return res


def quaternion_conjugate(Q):
        res = np.array([0,0,0,0])
        q = np.array(Q)
        res[0] = q[0]
        res[1:3] = -q[1:3]
        return res

def quaternion_rotation_z_pi(X):
    x = X
    z_ECI = -x/np.linalg.norm(x)
    vect = z_ECI*np.sin(m.pi/2)
    q = np.array([np.cos(np.pi/2), vect[0], vect[1], vect[2]])
    return q


def quaternion_rotation_y_piover2(X,V):
    x, v = X, V
    z_ECI = -x/np.linalg.norm(x)
    x_ECI = v/np.linalg.norm(v)
    y_ECI = np.cross(z_ECI, x_ECI)
    vect = y_ECI*np.sin(m.pi/4)
    return(np.array([np.cos(m.pi/4), vect[0], vect[1], vect[2]]))


def quaternion_multiplication(Q1,Q2):
    q1, q2 = np.array(Q1), np.array(Q2)
    vect = q1[0]*q2[1:4] + q2[0]*q1[1:4]+np.cross(q1[1:4],q2[1:4])
    return np.array([q1[0]*q2[0]-np.dot(q1,q2),vect[0],vect[1],vect[2]])


def nadir(X,V):
    x, v = X, V
    normx = np.linalg.norm(x)
    normv = np.linalg.norm(v)
    x = -x/normx
    v = v/normv

    c = -np.cross(x,v)

    M_SR = np.array([v,c,x])

    MRP = RigidBodyKinematics.C2MRP(M_SR) # Euler's axis of rotation for Nadir    
    return MRP, M_SR


def sun_sun(X,V,Sun_pos):
    x, v = X, V
    sun_pos = np.array(Sun_pos)
    sun_pos = sun_pos-x #satellite-sun vector
    Zs = sun_pos/np.linalg.norm(sun_pos)
    ps = np.dot(v,Zs)
    temp = ps*v
    Xs = temp-v
    Xs = Xs/np.linalg.norm(Xs)
    Ys = np.cross(Zs,Xs)
    M_R = np.array([Xs,Ys,Zs])
    Q = RigidBodyKinematics.C2EP(M_R) 
    return np.array(Q)


def sun_aero(X,V,sun_pos):
    x, v = X, V
    Sun_pos = sun_pos-x
    Xs = v/np.linalg.norm(v)
    print("dans sun_aero, la valeur de Xs est:")
    print(Xs)
    Zs = Sun_pos - np.dot(Sun_pos,Xs)*Xs
    Zs = Zs/np.linalg.norm(Zs)
    print("dans sun_aero, la valeur de Zs est:")
    print(Zs)
    Ys = np.cross(Zs,Xs)
    Ys = Ys/np.linalg.norm(Ys)
    print("dans sun_aero, la valeur de Ys est:")
    print(Ys)
    M_SR = np.array([Xs, Ys, Zs])
                
    Q = RigidBodyKinematics.C2EP(M_SR) # Euler's axis of rotation for Nadir
                
    return np.array(Q)

                        
def retrograde(X,V):
    x, v = X, V
    q1 = nadir(x, v)
    q2 = quaternion_rotation_z_pi(x)
    Q = quaternion_multiplication(q1,q2)

    return np.array(Q)


def drag(X,V):
    x, v = X, V
    q1 = quaternion_rotation_y_piover2(x,v)
    q2 = nadir(x,v)
    Q = quaternion_multiplication(q1,q2)

    return np.array(Q)

"""___________________________________________________________________________________________________________________________________"""

print("debut de run")
def run(show_plots):
    print("entré dans run")
    """
    The scenarios can be run with the followings setups parameters:

    Args:
        show_plots (bool): Determines if the script should display plots

    """

    #
    #  From here on scenario python code is found.  Above this line the code is to setup a
    #  unitTest environment.  The above code is not critical if learning how to code BSK.
    #

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"
    pyTaskName = "pyTask"
    pyProcessName = "pyProcess"
  
    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # set the simulation time variable used later on
    simulationTime = macros.min2nano(10.)

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName, 10)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(.1)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # create the process and task that contains the Python modules
    pyModulesProcess = scSim.CreateNewPythonProcess(pyProcessName, 9)
    pyModulesProcess.createPythonTask(pyTaskName, simulationTimeStep, True, -1)
    print("process et tasks créés")

    #
    #   setup the simulation tasks/objects
    #

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"

    # clear prior gravitational body and SPICE setup definitions
    gravFactory = simIncludeGravBody.gravBodyFactory()

    # setup Earth Gravity Body
    earth = gravFactory.createEarth()
    earth.isCentralBody = True  # ensure this is the central gravitational body
    mu = earth.mu

    # attach gravity model to spacecraft
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

    #
    #   initialize Spacecraft States with initialization variables
    #
    # setup the orbit using classical orbit elements
    oe = om.ClassicElements()
    oe.a = (6378 + 600)*1000.  # meters
    oe.e = 0.1
    oe.i = 63.3 * macros.D2R
    oe.Omega = 88.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 135.3 * macros.D2R
    rN, vN = om.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN  # m   - r_CN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_CN_N
    
    # define the simulation inertia
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]  # rad/s - omega_BN_B

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)
    print("module spacecraft ajouté")

    #Adding the sun position
    sunPositionMsgData = messaging.SpicePlanetStateMsgPayload()
    sunPositionMsgData.PositionVector = [0.0, om.AU*1000.0, 0.0] #This must be the position at a special time of the day where Y points toward the sun
    sun_pos = sunPositionMsgData.PositionVector

    

    #Build the derivative of the command quaternion
    #First, we have to determine which command quaternion we want based on the selected mode

    x = np.array(rN)
    print("la valeur de la position du SC est:")
    print(x)
    v = np.array(vN)
    print("la valeur de la vitesse du SC est:")
    print(v)
    
    #choice of the orientation we want
    m = 1

    MRPCmd = np.zeros(3)
    Matrix1 = np.zeros([3,3])

    #Nadir
    if(m==1):
        MRPCmd, Matrix1 = nadir(x,v)
    print("MRP nadir vaut:")
    print(MRPCmd)
    
    print("Matrix 1 vaut:")
    print(Matrix1)
    x = Matrix1[0]
    y = Matrix1[1]
    z = Matrix1[2]
    print("le produit scalaire de Z_nadir avec rN normé vaut:")
    rn = rN/np.linalg.norm(rN)
    print(np.dot(z,rn))
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])
    ax.set_zlim([-5,5])


    pts =5*rN/np.linalg.norm(rN)
    start = pts
    ax.quiver(start[0],start[1],start[2], x[0], x[1], x[2], color = 'c')
    ax.quiver(start[0],start[1],start[2], y[0], y[1], y[2], color = 'g')
    ax.quiver(start[0],start[1],start[2], z[0], z[1], z[2], color = 'r')
    ax.quiver(pts[0],pts[1],pts[2], rN[0], rN[1], rN[2], color = 'g')
    r = 0.05
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    a = np.cos(u) * np.sin(v)
    b = np.sin(u) * np.sin(v)
    c = np.cos(v)
    ax.plot_surface(a, b, c, cmap=plt.cm.YlGnBu_r)

    plt.show()
    """   
    #Sun_sun
    if(m==4):
        Q_Orientation = sun_sun(x,v,sun_pos)

    #Sun_aero
    if(m==5):
        Q_Orientation = sun_aero(x,v,sun_pos)

    #Retrograde
    if(m==2):
        Q_Orientation = retrograde(x,v) 
        
    #Drag
    if(m==3):
        Q_Orientation = drag(x,v) 

    print(Q_Orientation)
    """ 



    
    # setup extForceTorque module
    # the control torque is read in through the messaging system
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "externalDisturbance"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject)
    print("module extFobject ajouté")

    # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    # velocity navigation message
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)
    print("module sNavObject ajouté")
    #
    #   setup the FSW algorithm tasks
    #

# Here I can modify the desired orientation
    # setup inertial3D guidance module
    inertial3DConfig = inertial3D.inertial3DConfig()
    inertial3DWrap = scSim.setModelDataWrap(inertial3DConfig)
    inertial3DWrap.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DWrap, inertial3DConfig)
    #inertial3DConfig.sigma_R0N = [0. ,0. ,0. ]
    inertial3DConfig.sigma_R0N = [MRPCmd[0],MRPCmd[1],MRPCmd[2]]  # set the desired inertial orientation
    print("module inertial3DConfig ajouté")
  

    # setup the attitude tracking error evaluation module
    attErrorConfig = attTrackingError.attTrackingErrorConfig()
    attErrorWrap = scSim.setModelDataWrap(attErrorConfig)
    attErrorWrap.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attErrorWrap, attErrorConfig)
    print("module attErrorConfig ajouté")

    print("initialisation du module python")
  
    # setup Python MRP PD control module
    pyMRPPD = PythonMRPPD("pyMRP_PD", True, 100)
    pyMRPPD.K = 3.5
    pyMRPPD.P = 30.0
    pyModulesProcess.addModelToTask(pyTaskName, pyMRPPD)
    print("fin initialisation du module python")
    

    #   Setup data logging before the simulation is initialized
    #   
    #   Try to calculate here the vectors and record them
    numDataPoints = 50
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    attErrorLog = attErrorConfig.attGuidOutMsg.recorder(samplingTime)
    mrpLog = pyMRPPD.cmdTorqueOutMsg.recorder(samplingTime)
    scMRP = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, attErrorLog)
    scSim.AddModelToTask(simTaskName, mrpLog)
    scSim.AddModelToTask(simTaskName, scMRP)

    #
    # connect the messages to the modules  -  Travail a faire juste la
    #
    print("connection des modules")
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attErrorConfig.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attErrorConfig.attRefInMsg.subscribeTo(inertial3DConfig.attRefOutMsg)
    pyMRPPD.guidInMsg.subscribeTo(attErrorConfig.attGuidOutMsg)
    extFTObject.cmdTorqueInMsg.subscribeTo(pyMRPPD.cmdTorqueOutMsg)
    


    # if this scenario is to interface with the BSK Viz, uncomment the following lines
    vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                        # , saveFile=fileName
                                        )

    #
    #   initialize Simulation
    #
    scSim.InitializeSimulation()

    #
    #   configure a simulation stop time time and execute the simulation run
    #
    print("début sim")
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()
    print("fin sim")

   

    #simulationTime = macros.min2nano(20.)

    #scSim.ConfigureStopTime(simulationTime)
    #scSim.ExecuteSimulation()
    
    dataSC = scMRP.sigma_BN
    last_MRP = dataSC[-1]
    last_MRP_Euclidien = RigidBodyKinematics.MRP2C(last_MRP)

    x = last_MRP_Euclidien[0]
    y = last_MRP_Euclidien[1]
    z = last_MRP_Euclidien[2]

    intial_Orientation = RigidBodyKinematics.MRP2C(inertial3DConfig.sigma_R0N)
    xi = intial_Orientation[0]
    yi = intial_Orientation[1]
    zi = intial_Orientation[2]
    print("le produit scalaire de Z_nadir avec rN normé vaut:")
    rn = rN/np.linalg.norm(rN)
    print(np.dot(z,rn))
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])
    ax.set_zlim([-5,5])

    pts =[0,0,0]
    start = pts
    ax.quiver(start[0],start[1],start[2], x[0], x[1], x[2], color = 'b')
    ax.quiver(start[0],start[1],start[2], y[0], y[1], y[2], color = 'g')
    ax.quiver(start[0],start[1],start[2], z[0], z[1], z[2], color = 'r')
    ax.quiver(start[0],start[1],start[2], xi[0], xi[1], xi[2], color = 'c')
    ax.quiver(start[0],start[1],start[2], yi[0], yi[1], yi[2], color = 'c')
    ax.quiver(start[0],start[1],start[2], zi[0], zi[1], zi[2], color = 'c')
    
   
    plt.show()


    #
    #   retrieve the logged data
    #
    dataLr = mrpLog.torqueRequestBody
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
    timeAxis = attErrorLog.times()
    np.set_printoptions(precision=16)

    #Verif
    print(dataLr)
    print(dataSigmaBR)
    print(dataOmegaBR)

    #
    #   plot the results
    #
    plt.close("all")  # clears out plots from earlier test runs
    plt.figure(1)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataSigmaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')
    figureList = {}
    pltName = fileName + "1"
    figureList[pltName] = plt.figure(1)

    plt.figure(2)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataLr[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$L_{r,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Control Torque $L_r$ [Nm]')
    pltName = fileName + "2"
    figureList[pltName] = plt.figure(2)

    plt.figure(3)
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error [rad/s] ')

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return figureList




class PythonMRPPD(simulationArchTypes.PythonModelClass):
    """
    This class inherits from the `PythonModelClass` available in the ``simulationArchTypes`` module.
    The `PythonModelClass` is the parent class which your Python BSK modules must inherit.
    The class uses the following
    virtual functions:

    #. ``reset``: The method that will initialize any persistent data in your model to a common
       "ready to run" state (e.g. filter states, integral control sums, etc).
    #. ``updateState``: The method that will be called at the rate specified
       in the PythonTask that was created in the input file.

    Additionally, your class should ensure that in the ``__init__`` method, your call the super
    ``__init__`` method for the class so that the base class' constructor also gets called to
    initialize the model-name, activity, moduleID, and other important class members:

    .. code-block:: python

        super(PythonMRPPD, self).__init__(modelName, modelActive, modelPriority)

    You class must implement the above four functions. Beyond these four functions you class
    can complete any other computations you need (``Numpy``, ``matplotlib``, vision processing
    AI, whatever).
    """
    def __init__(self, modelName, modelActive=True, modelPriority=-1):
        super(PythonMRPPD, self).__init__(modelName, modelActive, modelPriority)

        # Proportional gain term used in control
        self.K = 0
        # Derivative gain term used in control
        self.P = 0
        # Input guidance structure message
        self.guidInMsg = messaging.AttGuidMsgReader()
        # Output body torque message name
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

    def reset(self, currentTime):
        """
        The reset method is used to clear out any persistent variables that need to get changed
        when a task is restarted.  This method is typically only called once after selfInit/crossInit,
        but it should be written to allow the user to call it multiple times if necessary.
        :param currentTime: current simulation time in nano-seconds
        :return: none
        """
        return


    def updateState(self, currentTime):
        """
        The updateState method is the cyclical worker method for a given Basilisk class.  It
        will get called periodically at the rate specified in the Python task that the model is
        attached to.  It persists and anything can be done inside of it.  If you have realtime
        requirements though, be careful about how much processing you put into a Python updateState
        method.  You could easily detonate your sim's ability to run in realtime.

        :param currentTime: current simulation time in nano-seconds
        :return: none
        """
        # read input message
        guidMsgBuffer = self.guidInMsg()

        # create output message buffer
        torqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()

        # compute control solution
        lrCmd = np.array(guidMsgBuffer.sigma_BR) * self.K + np.array(guidMsgBuffer.omega_BR_B) * self.P
        torqueOutMsgBuffer.torqueRequestBody = (-lrCmd).tolist()

        self.cmdTorqueOutMsg.write(torqueOutMsgBuffer, currentTime, self.moduleID)

        def print_output():
            """Sample Python module method"""
            print(currentTime * 1.0E-9)
            print(torqueOutMsgBuffer.torqueRequestBody)
            print(guidMsgBuffer.sigma_BR)
            print(guidMsgBuffer.omega_BR_B)

        return








#
# This statement below ensures that the unit test scrip can be run as a
# stand-along python script
#
if __name__ == "__main__":
    run(
        True  # show_plots
    )









