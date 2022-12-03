# import simulation related support
from Basilisk.simulation import spacecraft
# general support file with common unit test functions
# import general simulation support files
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody, unitTestSupport, vizSupport,astroConstants, rigidBodyDynamics, planetStates)

import math as m
import numpy as np



def quaternion_simulation(i, time): #i in  [1, 8]

     """
    The scenarios can be run with the followings setups parameters:

    Args:
        show_plots (bool): Determines if the script should display plots
        planetCase (string): Specify if a `Mars` or `Earth` arrival is simulated

    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName)


    scObject = spacecraft.Spacecraft()
    speed = scObject.v_CN_NInit
    position = scObject.r_CN_NInit
    sun_p, sun_v = planetStates.planetPositionVelocity(Sun, time, ephemerisPath='/supportData/EphemerisData/pck00010.tpc', observer='SSB', frame='J2000') #OU TROUVER CA --> REGARDER SUR BASILISK

    if(i=1):
        q_ref = orbit_quaternion(speed, position)
        state = 0
    if(i=5):
        q1 = orbit_quaternion(speed, position)
        q2 = quaternion_rotation_z_pi(position)
        q_ref = q_multiplication(q1,q2)
        state = 0
    if(i=4):
        q1 = orbit_quaternion(speed, position)
        q2 = quaternion_rotation_y_pi_05(position)
        q_ref = q_multiplication(q1,q2)
        state = 0
    if(i=3):
        q_ref = quaternion_sun_sun(sun_p, speed)
        state = 0
    if(i=2):
        q_ref = quaternion_sun_aero(sun_p, speed)
        state = 0
    if(i=8): #Stands for the 'default' case in MatLab
        q_ref = orbit_quaternion(speed, position)
        state = 0
    if(i=6):
        q_ref = np.array([1,0,0,0])
        state = 1
    if(i=7):
        q_ref = np.array([1,0,0,0])
        state = 0

    return(q_ref, state)

if __name__ == '__main__':
    quaternion_simulation(i, time) #How do we enter i & time ?



#################################### fonctions ############################################
def orbit_quaternion(speed, position):
    u_speed = speed/m.sqrt(np.dot(speed, speed))
    u_position = position/m.sqrt(np.dot(position, position))
    cross_product = np.cross(u_speed, u_position)
    m = np.array(u_speed, -cross_product, u_position)
    q = RigidBodyKinematics.C2EP(m)
    q_inv = np.array([q[0],-q[1],-q[2],-q[3])
    return q_inv


def quaternion_rotation_z_pi(position):
    return(np.array([0,position[0], position[1], position[2]))

   
def quaternion_rotation_y_pi_05(position, vitesse):
    cross_product = sin(m.pi/4)*np.cross(position, vitesse)
    q = np.array([cos(pi/4),cross_product[0],cross_product[1],cross_product[2]])
    return q


def quaternion_sun_sun(sun_p, speed):
    u_sun_p = sun_p/np.sqrt(np.dot(sun_p,sun_p))
    scal1 = np.dot(speed, u_sun_p)
    vect1 = scal1*u_sun_p
    vect2 = vect1 - speed
    vect2 = vect2/np.sqrt(np.dot(vect2,vect2))
    vect3 = np.cross(u_sun_p, vect2)
    matrice = np.array([vect2[0],vect2[1],vect2[2]],
                       [vect3[0],vect3[1],vect3[2]],
                       [u_sun_p[0],u_sun_p[1],u_sun_p[2]])
    q = RigidBodyKinematics.C2EP(matrice)
    q_inv = np.array([q[O], -q[1], -q[2], -q[3]])
    return q_inv


def quaternion_sun_aero(sun_p, speed):
    u_speed = speed/np.sqrt(np.dot(speed,speed))
    scal1 = np.dot(u_speed, sun_p)
    vect1 = scal1*u_speed
    vect2 = sun_p - vect1
    vect2 = vect2/np.sqrt(np.dot(vect2,vect2))
    vect3 = np.cross(vect2, u_speed)
    matrice = np.array([u_speed[0],u_speed[1],u_speed[2]],
                       [vect3[0],vect3[1],vect3[2]],
                       [vect2[0],vect2[1],vect2[2]])
    q = RigidBodyKinematics.C2EP(matrice)
    q_inv = np.array([q[O], -q[1], -q[2], -q[3]])
    return q_inv



################################fonctions auxiliaires#####################################
def norme_quaternion(q):
    return m.sqrt(q[0]*2+q[1]*2+q[2]*2+q[3]*2)

def q_multiplication(q1, q2):
    vect = q1[0]*q2[1,3] + q2[0]*q1[1,3]+np.cross(q1[1,3],q2[1,3])
    return np.array([q1[0]*q2[0]-np.dot(q1,q2),vect[0],vect[1],vect[2])
   
