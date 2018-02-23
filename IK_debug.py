from sympy import sqrt, atan2, acos, symbols, pi, cos, sin, simplify, sympify
from time import time
from mpmath import radians
import tf
import numpy as np

from sympy.matrices import Matrix
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3

q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

dh_params = {
    alpha0:     0, a0:      0, d1:  0.75,
    alpha1: -pi/2, a1:   0.35, d2:     0, q2: q2 - pi/2,
    alpha2:     0, a2:   1.25, d3:     0,
    alpha3: -pi/2, a3: -0.054, d4:   1.5,
    alpha4:  pi/2, a4:      0, d5:     0,
    alpha5: -pi/2, a5:      0, d6:     0,
    alpha6:     0, a6:      0, d7: 0.303, q7: 0 }


def dh_transform(alpha, a, theta, d):
    return Matrix([[           cos(theta),           -sin(theta),           0,             a],
                   [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                   [sin(theta)*sin(alpha), cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                   [                    0,                     0,           0,             1]])

T0_1 = dh_transform(alpha0, a0, q1, d1).subs(dh_params)
T1_2 = dh_transform(alpha1, a1, q2, d2).subs(dh_params)
T2_3 = dh_transform(alpha2, a2, q3, d3).subs(dh_params)
T3_4 = dh_transform(alpha3, a3, q4, d4).subs(dh_params)
# T4_5 = dh_transform(alpha4, a4, q5, d5).subs(dh_params)
# T5_6 = dh_transform(alpha5, a5, q6, d6).subs(dh_params)
# T6_G = dh_transform(alpha6, a6, q7, d7).subs(dh_params)

# T0_2 = simplify(T0_1 * T1_2)
# T0_3 = simplify(T0_2 * T2_3)
# T0_4 = simplify(T0_3 * T3_4)
# T0_5 = simplify(T0_4 * T4_5)
# T0_6 = simplify(T0_5 * T5_6)
# T0_G = simplify(T0_6 * T6_G)
T0_2 = T0_1 * T1_2
T0_3 = T0_2 * T2_3
T0_4 = T0_3 * T3_4
# T0_5 = T0_4 * T4_5
# T0_6 = T0_5 * T5_6
# T0_G = simplify(T0_6 * T6_G)


# numpy version of the simplified T0_G
def np_T0_G(q1, q2, q3, q4, q5, q6):
  (q1, q2, q3, q4, q5, q6) = (float(q1), float(q2), float(q3), float(q4), float(q5), float(q6))
  return np.array([
    [((np.sin(q1)*np.sin(q4) + np.sin(q2 + q3)*np.cos(q1)*np.cos(q4))*np.cos(q5) + np.sin(q5)*np.cos(q1)*np.cos(q2 + q3))*np.cos(q6) - (-np.sin(q1)*np.cos(q4) + np.sin(q4)*np.sin(q2 + q3)*np.cos(q1))*np.sin(q6), -((np.sin(q1)*np.sin(q4) + np.sin(q2 + q3)*np.cos(q1)*np.cos(q4))*np.cos(q5) + np.sin(q5)*np.cos(q1)*np.cos(q2 + q3))*np.sin(q6) + (np.sin(q1)*np.cos(q4) - np.sin(q4)*np.sin(q2 + q3)*np.cos(q1))*np.cos(q6), -(np.sin(q1)*np.sin(q4) + np.sin(q2 + q3)*np.cos(q1)*np.cos(q4))*np.sin(q5) + np.cos(q1)*np.cos(q5)*np.cos(q2 + q3), -0.303*np.sin(q1)*np.sin(q4)*np.sin(q5) + 1.25*np.sin(q2)*np.cos(q1) - 0.303*np.sin(q5)*np.sin(q2 + q3)*np.cos(q1)*np.cos(q4) - 0.054*np.sin(q2 + q3)*np.cos(q1) + 0.303*np.cos(q1)*np.cos(q5)*np.cos(q2 + q3) + 1.5*np.cos(q1)*np.cos(q2 + q3) + 0.35*np.cos(q1)],
    [ ((np.sin(q1)*np.sin(q2 + q3)*np.cos(q4) - np.sin(q4)*np.cos(q1))*np.cos(q5) + np.sin(q1)*np.sin(q5)*np.cos(q2 + q3))*np.cos(q6) - (np.sin(q1)*np.sin(q4)*np.sin(q2 + q3) + np.cos(q1)*np.cos(q4))*np.sin(q6), -((np.sin(q1)*np.sin(q2 + q3)*np.cos(q4) - np.sin(q4)*np.cos(q1))*np.cos(q5) + np.sin(q1)*np.sin(q5)*np.cos(q2 + q3))*np.sin(q6) - (np.sin(q1)*np.sin(q4)*np.sin(q2 + q3) + np.cos(q1)*np.cos(q4))*np.cos(q6), -(np.sin(q1)*np.sin(q2 + q3)*np.cos(q4) - np.sin(q4)*np.cos(q1))*np.sin(q5) + np.sin(q1)*np.cos(q5)*np.cos(q2 + q3),  1.25*np.sin(q1)*np.sin(q2) - 0.303*np.sin(q1)*np.sin(q5)*np.sin(q2 + q3)*np.cos(q4) - 0.054*np.sin(q1)*np.sin(q2 + q3) + 0.303*np.sin(q1)*np.cos(q5)*np.cos(q2 + q3) + 1.5*np.sin(q1)*np.cos(q2 + q3) + 0.35*np.sin(q1) + 0.303*np.sin(q4)*np.sin(q5)*np.cos(q1)],
    [                                                                -(np.sin(q5)*np.sin(q2 + q3) - np.cos(q4)*np.cos(q5)*np.cos(q2 + q3))*np.cos(q6) - np.sin(q4)*np.sin(q6)*np.cos(q2 + q3),                                                     (np.sin(q5)*np.sin(q2 + q3) - np.cos(q4)*np.cos(q5)*np.cos(q2 + q3))*np.sin(q6) - np.sin(q4)*np.cos(q6)*np.cos(q2 + q3),           -np.sin(q5)*np.cos(q4)*np.cos(q2 + q3) - np.sin(q2 + q3)*np.cos(q5),                                                                                 -0.303*np.sin(q5)*np.cos(q4)*np.cos(q2 + q3) - 0.303*np.sin(q2 + q3)*np.cos(q5) - 1.5*np.sin(q2 + q3) + 1.25*np.cos(q2) - 0.054*np.cos(q2 + q3) + 0.75],
    [                                                                                                                                                            0,                                                                                                                                                0,                                                              0,                                                                                                  1]])

def np_R0_3(q1, q2, q3):
  (q1, q2, q3) = (float(q1), float(q2), float(q3))
  return np.array([[np.sin(q2 + q3)*np.cos(q1), np.cos(q1)*np.cos(q2 + q3), -np.sin(q1)],
                   [np.sin(q1)*np.sin(q2 + q3), np.sin(q1)*np.cos(q2 + q3),  np.cos(q1)],
                   [        np.cos(q2 + q3),        -np.sin(q2 + q3),        0]])

# R0_3 = simplify(T0_1[0:3,0:3]*T1_2[0:3,0:3]*T2_3[0:3,0:3])

def Rot_X(angle):
    return rot_axis1(angle).transpose()

def Rot_Y(angle):
    return rot_axis2(angle).transpose()

def Rot_Z(angle):
    return rot_axis3(angle).transpose()

#R_corr = simplify(Rot_Z(pi) * Rot_Y(-pi/2))

# simpler non parametric version
R_corr = np.array([[0,  0, 1],
                   [0, -1, 0],
                   [1,  0, 0]])

'''
Format of test case is [ [[EE position],[EE orientation as quaternions]],[WC location],[joint angles]]
You can generate additional test cases by setting up your kuka project and running `$ roslaunch kuka_arm forward_kinematics.launch`
From here you can adjust the joint angles to find thetas, use the gripper to extract positions and orientation (in quaternion xyzw) and lastly use link 5
to find the position of the wrist center. These newly generated test cases can be added to the test_cases dictionary.
'''

test_cases = {1:[[[2.16135,-1.42635,1.55109],
                  [0.708611,0.186356,-0.157931,0.661967]],
                  [1.89451,-1.44302,1.69366],
                  [-0.65,0.45,-0.36,0.95,0.79,0.49]],
              2:[[[-0.56754,0.93663,3.0038],
                  [0.62073, 0.48318,0.38759,0.480629]],
                  [-0.638,0.64198,2.9988],
                  [-0.79,-0.11,-2.33,1.94,1.14,-3.68]],
              3:[[[-1.3863,0.02074,0.90986],
                  [0.01735,-0.2179,0.9025,0.371016]],
                  [-1.1669,-0.17989,0.85137],
                  [-2.99,-0.12,0.94,4.06,1.29,-4.12]],
              4:[],
              5:[]}

class JointsOption:
    LIMITS = [[-3.23, 3.23],[-0.79, 1.48], [-3.67, 1.13], [-6.11, 6.11], [-2.18, 2.18], [-6.11, 6.11]]
    def __init__(self):
        self.theta1 = 0
        self.theta2 = 0
        self.theta3 = 0
        self.theta4 = 0
        self.theta5 = 0
        self.theta6 = 0
        self.wx_tilde = 0
        self.wz_tilde = 0
        self.valid = True

    def clone(self):
        clone = JointsOption()
        clone.theta1 = self.theta1
        clone.theta2 = self.theta2
        clone.theta3 = self.theta3
        clone.theta4 = self.theta4
        clone.theta5 = self.theta5
        clone.theta6 = self.theta6
        clone.wx_tilde = self.wx_tilde
        clone.wz_tilde = self.wz_tilde
        clone.valid = self.valid
        return clone

    def thetas(self):
        return (self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6)

    def wc_error(self, wc):
        fk_wc = T0_4.evalf(subs={q1:self.theta1, q2:self.theta2, q3:self.theta3, q4:self.theta4, q5:self.theta5, q6:self.theta6})[:3,3]
        wc_x_e = abs(fk_wc[0]-wc[0])
        wc_y_e = abs(fk_wc[1]-wc[1])
        wc_z_e = abs(fk_wc[2]-wc[2])
        return sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)

    def ee_pos(self, local_T0_G=None):
        if local_T0_G is not None:
            return local_T0_G[:3,3]
        else:                
            return np_T0_G(self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6)[:3,3]

    def ee_orientation(self, local_T0_G=None):
        corr = Matrix.eye(4)
        corr[:3,:3] = R_corr
        if local_T0_G is not None:
            rot_mat = local_T0_G
        else:
            rot_mat = np_T0_G(self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6)
        rot_mat = rot_mat * corr        
        return tf.transformations.quaternion_from_matrix(rot_mat.tolist())

    def ee_error(self, ee, local_T0_G=None):
        fk_ee = self.ee_pos(local_T0_G)
        ee_x_e = abs(fk_ee[0]-ee[0])
        ee_y_e = abs(fk_ee[1]-ee[1])
        ee_z_e = abs(fk_ee[2]-ee[2])
        return sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)

    def orientation_error(self, orientation, local_T0_G=None):                
        fk_orientation = self.ee_orientation(local_T0_G)
        err = np.linalg.norm(fk_orientation - np.array([orientation.x, orientation.y, orientation.z, orientation.w]))
        return err

    def fk_error(self, ee, orientation, wc=None):    
        local_T0_G = np_T0_G(self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6)
        return self.ee_error(ee, local_T0_G) + self.orientation_error(orientation, local_T0_G) #+ self.wc_error(wc)        


    def check_valid(self):
        if not self.valid:
            return False
        thetas = self.thetas()
        for i in range(len(thetas)):
            if not sympify(thetas[i]).is_real:
                return False
            if not (JointsOption.LIMITS[i][0] <= thetas[i] and thetas[i] <= JointsOption.LIMITS[i][1]):
                # print ("theta{}".format(i), thetas[i].evalf(), "not in limits", JointsOption.LIMITS[i])
                return False
        return True

    def debug_print(self):
        print([t.evalf() for t in self.thetas()])

def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0
    class Position:
        def __init__(self,EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]
    class Orientation:
        def __init__(self,EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self,position,orientation):
            self.position = position
            self.orientation = orientation

    comb = Combine(position,orientation)

    class Pose:
        def __init__(self,comb):
            self.poses = [comb]

    req = Pose(comb)
    start_time = time()

    ########################################################################################
    ##

    ## Insert IK code here!

    px = req.poses[x].position.x
    py = req.poses[x].position.y
    pz = req.poses[x].position.z

    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [req.poses[x].orientation.x, req.poses[x].orientation.y,
            req.poses[x].orientation.z, req.poses[x].orientation.w])

    ### Your IK code here
    # Compensate for rotation discrepancy between DH parameters and Gazebo
    #
    #
    Rrpy = Rot_Z(yaw) * Rot_Y(pitch) * Rot_X(roll) * R_corr

    # Calculate wrist center position
    wx, wy, wz = Matrix([px, py, pz]) - (dh_params[d6] + dh_params[d7]) * Rrpy[:,2]

    # Calculate joint angles using Geometric IK method
    #
    #
    ###
    joints = []
    joints.append(JointsOption())
    # joints.append(JointsOption())
    joints[0].theta1 = atan2(wy, wx)
    # joints[1].theta1 = atan2(-wy,-wx) # 2nd_alternative    

    # Calculate theta2 and theta3 from triangle spanned by joint2, joint3 and wrist-center
    joints[0].wx_tilde = sqrt(wx * wx + wy * wy) - dh_params[a1] # 1st_alternative
    joints[0].wz_tilde = wz - dh_params[d1]
    # joints[1].wx_tilde = -sqrt(wx * wx + wy * wy) - dh_params[a1] # 2nd_alternative
    # joints[1].wz_tilde = wz - dh_params[d1]

    joints = [j for j in joints if j.check_valid()]

    joints_alt = []
    for j in joints:

        len_A = sqrt(dh_params[d4] * dh_params[d4] + dh_params[a3] * dh_params[a3])
        len_B = sqrt(j.wx_tilde * j.wx_tilde + j.wz_tilde * j.wz_tilde)
        len_C = dh_params[a2]
        if len_B > len_A + len_C:
            j.valid = False
            continue

        j_alt = j.clone()

        j.theta2 = atan2(j.wx_tilde, j.wz_tilde) - acos((len_B*len_B + len_C*len_C - len_A*len_A)/(2*len_B*len_C)) # 1nd_alternative
        j.theta3 = pi/2 - acos((len_A*len_A + len_C*len_C - len_B*len_B)/(2*len_A*len_C)) + atan2(dh_params[a3], dh_params[d4]) # 1st_alternative

        j_alt.theta2 = atan2(j.wx_tilde, j.wz_tilde) + acos((len_B*len_B + len_C*len_C - len_A*len_A)/(2*len_B*len_C)) # 2nd_alternative
        j_alt.theta3 = -3*pi/2 + acos((len_A*len_A + len_C*len_C - len_B*len_B)/(2*len_A*len_C)) - atan2(dh_params[a3], dh_params[d4]) # 2nd_alternative
        joints_alt.append(j_alt)

    joints = [j for j in (joints + joints_alt) if j.check_valid()]
    joints_alt = []
    for j in joints:        
        R0_3_current = np_R0_3(j.theta1, j.theta2,  j.theta3)
        # R0_3_current = R0_3.evalf(subs={q1:j.theta1, q2:j.theta2, q3: j.theta3})
        #R3_6 = R0_3_current.inv("LU") * Rrpy
        R3_6 = R0_3_current.T * Rrpy

        # R3_6(q4, q5, q6) structure:
        # [r00, r01, r02]
        # [r10, r11, r12]
        # [r20, r21, r22]
        #
        #[[-sin(q4)*sin(q6) + cos(q4)*cos(q5)*cos(q6), -sin(q4)*cos(q6) - sin(q6)*cos(q4)*cos(q5), -sin(q5)*cos(q4)],
        # [                           sin(q5)*cos(q6),                           -sin(q5)*sin(q6),          cos(q5)],
        # [-sin(q4)*cos(q5)*cos(q6) - sin(q6)*cos(q4),  sin(q4)*sin(q6)*cos(q5) - cos(q4)*cos(q6),  sin(q4)*sin(q5)]]
        #
        # theta = atan2(sin(theta), cos(theta))

        j.theta4 = atan2(R3_6[2,2], -R3_6[0,2])
        j.theta5 = atan2(sqrt(R3_6[0,2] * R3_6[0,2] + R3_6[2,2] * R3_6[2,2] ), R3_6[2,2])
        j.theta6 = atan2(-R3_6[1,1], R3_6[1,0])

        j_alt = j.clone()
        j_alt.theta5 = (acos(R3_6[1,2])) # ambiguous but numerically more accurate alternative
        joints_alt.append(j_alt)

        j_alt = j.clone()
        j_alt.theta6 = atan2(R3_6[1,1], -R3_6[1,0])
        joints_alt.append(j_alt)

        j_alt = j.clone()        
        j_alt.theta4 = atan2(-R3_6[2,2], R3_6[0,2])
        joints_alt.append(j_alt)

        j_alt = j.clone()        
        j_alt.theta4 = atan2(-R3_6[2,2], R3_6[0,2])
        j_alt.theta6 = atan2(R3_6[1,1], -R3_6[1,0])
        joints_alt.append(j_alt)        


    joints = [j for j in (joints + joints_alt) if j.check_valid()]
    # for j in joints:
    #     j.debug_print()
    #     print(j.fk_error([px, py, pz], req.poses[x].orientation))

    # print(len(joints))
    joints = sorted(joints, key = lambda j: j.fk_error([px, py, pz], req.poses[x].orientation))
    (theta1, theta2, theta3, theta4, theta5, theta6) = joints[0].thetas()
    # print(test_case)
    # print(theta1.evalf(), theta2.evalf(), theta3.evalf(), theta4.evalf(), theta5.evalf(), theta6.evalf())
    ##
    ########################################################################################

    ########################################################################################
    ## For additional debugging add your forward kinematics here. Use your previously calculated thetas
    ## as the input and output the position of your end effector as your_ee = [x,y,z]

    ## (OPTIONAL) YOUR CODE HERE!

    ## End your code input for forward kinematics here!
    ########################################################################################

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    #your_wc = [wx,wy,wz] # <--- Load your calculated WC values in this array
    #your_ee = [1,1,1] # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    your_wc = T0_4.evalf(subs={q1:theta1, q2: theta2, q3:theta3, q4:theta4, q5: theta5, q6: theta6})[:3,3]
    your_ee = joints[0].ee_pos()

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time()-start_time))

    # Find WC error
    if not(sum(your_wc)==3):
        wc_x_e = abs(your_wc[0]-test_case[1][0])
        wc_y_e = abs(your_wc[1]-test_case[1][1])
        wc_z_e = abs(your_wc[2]-test_case[1][2])
        wc_offset = sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)
        print ("\nWrist error for x position is: %04.8f" % wc_x_e)
        print ("Wrist error for y position is: %04.8f" % wc_y_e)
        print ("Wrist error for z position is: %04.8f" % wc_z_e)
        print ("Overall wrist offset is: %04.8f units" % wc_offset)

    # Find theta errors
    t_1_e = abs(theta1-test_case[2][0])
    t_2_e = abs(theta2-test_case[2][1])
    t_3_e = abs(theta3-test_case[2][2])
    t_4_e = abs(theta4-test_case[2][3])
    t_5_e = abs(theta5-test_case[2][4])
    t_6_e = abs(theta6-test_case[2][5])
    print ("\nTheta 1 error is: %04.8f" % t_1_e)
    print ("Theta 2 error is: %04.8f" % t_2_e)
    print ("Theta 3 error is: %04.8f" % t_3_e)
    print ("Theta 4 error is: %04.8f" % t_4_e)
    print ("Theta 5 error is: %04.8f" % t_5_e)
    print ("Theta 6 error is: %04.8f" % t_6_e)
    print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
           \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
           \nconfirm whether your code is working or not**")
    print (" ")

    # Find FK EE error
    if not(sum(your_ee)==3):
        ee_x_e = abs(your_ee[0]-test_case[0][0][0])
        ee_y_e = abs(your_ee[1]-test_case[0][0][1])
        ee_z_e = abs(your_ee[2]-test_case[0][0][2])
        ee_offset = sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)
        print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
        print ("End effector error for y position is: %04.8f" % ee_y_e)
        print ("End effector error for z position is: %04.8f" % ee_z_e)
        print ("Overall end effector offset is: %04.8f units \n" % ee_offset)


extra_test_cases = [
    [[[2.0429658999709512, 0.00018577714877130244, 1.9460065037129493],
      [1.6915786841868415e-05, 2.2216975901156292e-05, 4.54682863298307e-05, 0.9999999985764485]],
      [0,0,0],
      [9.09346206006459e-5, -0.0882762556917320, 0.0850182806134346, 8.18731437693444e-7, 0.00330240749180236, 3.30148669177341e-5]],

    [[[2.0195262606622473, 0.16698506462990406, 2.0417529615877315],
      [0.0003220130175614815, -0.03951771657341912, 0.042890118940066065, 0.9982978934575323]],
      [0,0,0],
      [0.0818871376364666, -0.101301418307218, 0.0491628061655177, 2.99080046075358, 0.0271810482099740, -2.99328757937539]],

    [[[1.9769441960971719, 0.33048109903625733, 2.136772764172124],
      [0.00349678024873619, -0.07808266781966813, 0.08910823355251568, 0.992950448022058]],
      [0,0,0],
      [0.163192895963140, -0.114392202466146, 0.0132244651565888, 2.85647851187601, 0.0574925723473332, -2.86143440611162]],

    [[[1.903007344673762, 0.5194399889429554, 2.2461790612857033],
      [0.010442307454999281, -0.12116982972366755, 0.1488362396764106, 0.9813544743561007]],
      [0,0,0],
      [0.260068658728734, -0.129094597984821, -0.0295035340808737, 2.68488192426359, 0.0945860391592334, -2.69235126743959]],

    [[[1.8042447924864264, 0.6962914916955666, 2.3525704356014896],
      [0.01961216320332024, -0.1596319596434577, 0.21165715676180313, 0.9640198382327956]],
      [0,0,0],
      [0.356270381057115, -0.143905556044535, -0.0727567344822062, 2.51172693206139, 0.130811941018764, -2.52150422363392]],

    [[[1.6413968921409114, 0.9088546774476871, 2.4835190770471973],
      [0.03299557252938887, -0.20001280997450654, 0.2989053671087394, 0.9325029488174016]],
      [0,0,0],
      [0.483520373532784, -0.161999795737640, -0.130547307694652, 2.26315880120854, 0.179251307726778, -2.27504777305222]],

    [[[1.4471492122688707, 1.086466364630623, 2.6059639230593925],
      [0.04519436396921005, -0.22751467794687305, 0.3845243651457539, 0.893496252590577]],
      [0,0,0],
      [0.610147143778954, -0.180348245981079, -0.189961191502288, 2.01024070738093, 0.225190345339878, -2.02369376140177]],

    [[[1.204155873245472, 1.2388644160777487, 2.7309987212595996],
      [0.05506249893098041, -0.2420501260257162, 0.47370251360523247, 0.8449767963124759]],
      [0,0,0],
      [0.752543589795000, -0.200645630375664, -0.258470889150799, 1.71390127803139, 0.274435388666447, -1.72832717628654]],

    [[[0.9483013095537092, 1.3359834894194502, 2.8449353224017626],
      [0.06087989629969198, -0.2421955800773722, 0.5493977738886151, 0.7973688138269255]],
      [0,0,0],
      [0.895939715254021, -0.221353905478895, -0.328753326547168, 1.41007634708051, 0.320659608856086, -1.42519028657576]],

    [[[0.794085292276041, 1.3965762398927533, 2.8878956924408747],
      [0.05495502463843831, -0.2136478583920307, 0.5930101970686397, 0.7743858495892733]],
      [0,0,0],
      [0.994208794017649, -0.224329190819271, -0.373159269246273, 1.10776612633645, 0.360641434835303, -1.12095470946985]],

    [[[0.6436790179817689, 1.4361730485426496, 2.9288745731611234],
      [0.049951785331388936, -0.18038389881470274, 0.6213560127818699, 0.7608437248019009]],
      [0,0,0],
      [1.09558726738547, -0.227344471136412, -0.417991913664522, 0.799312199990472, 0.400698141167449, -0.811770926121417]],

    [[[0.5254279408106813, 1.45463402562161, 2.9614034802616627],
      [0.04936339824883543, -0.15146995296715857, 0.632573751561225, 0.7579383596950422]],
      [0,0,0],
      [1.18192692066014, -0.229265125510437, -0.454419078779885, 0.536904314497009, 0.435040061785694, -0.549937610080384]],

    [[[0.41264423979724074, 1.4595216226612675, 2.9952713968178033],
      [0.055332951913532616, -0.12473870803760176, 0.6333450206918126, 0.7617431351279328]],
      [0,0,0],
      [1.27093873862325, -0.231080497374426, -0.489950543705194, 0.271723735284524, 0.470801229134029, -0.286739701510859]],

    [[[0.35265319388143324, 1.4597005363028281, 3.012855319999201],
      [0.062368712434898464, -0.10912708643998253, 0.6280786344326044, 0.767931410793704]],
      [0,0,0],
      [1.32337543049106, -0.230730394412089, -0.509185382871681, 0.108700139830443, 0.493546844859799, -0.125365716872923]],

    [[[0.29455308294461474, 1.4556414825776414, 3.0315649914151703],
      [0.07326717534334663, -0.095660549149184, 0.6196709801421635, 0.775557126666588]],
      [0,0,0],
      [1.37663837156471, -0.230288208585469, -0.527801854329233, -0.0545450628119869, 0.517114997851362, 0.0357258341949668]],

    [[[0.26056756337264686, 1.453896897411386, 3.0416790128054383],
      [0.08224669638575909, -0.0869798310969896, 0.6114294415680245, 0.7821918101717644]],
      [0,0,0],
      [1.41067487236994, -0.228794621396409, -0.538618798557758, -0.165731055380918, 0.533660991815693, 0.145465446636331]],

    [[[0.22716838281421203, 1.4504966311226761, 3.0524558457115836],
      [0.09326307200639644, -0.0796638139993489, 0.6020467345654359, 0.78898378027568]],
      [0,0,0],
      [1.44494678923674, -0.227242563441535, -0.549093174221875, -0.276700940917365, 0.550619486095329, 0.254835293510992]],

    [[[0.22941499229671966, 1.454772488923645, 3.0492495079951367],
      [0.09338177093337077, -0.07780239927447596, 0.5996930927952283, 0.790945526555512]],
      [0,0,0],
      [1.44513630704576, -0.225557588952875, -0.548261563174589, -0.288750250382477, 0.552480372446703, 0.267108669676669]],

    [[[0.2316638076938784, 1.4590279769338825, 3.046043344546694],
      [0.0935209893124421, -0.07597145041721176, 0.5973198573209936, 0.7929004674797179]],
      [0,0,0],
      [1.44532284820327, -0.223872125290387, -0.547425063328515, -0.300782977409614, 0.554344206851292, 0.279368750019292]],

    [[[0.23391464469568657, 1.463263137608305, 3.0428374959769604],
      [0.09368069788433277, -0.07417128333360407, 0.5949274142959657, 0.7948483624513967]],
      [0,0,0],
      [1.44550636695431, -0.222186174952197, -0.546583653070526, -0.312799227392757, 0.556210887353843, 0.291615653582512]],

    [[[0.23616731848470526, 1.4674780160559966, 3.0396321014204966],
      [0.093860864566541, -0.07240220910867855, 0.5925161549414596, 0.796788971028335]],
      [0,0,0],
      [1.44568681791495, -0.220499740508202, -0.545737311549291, -0.324799108338157, 0.558080313073323, 0.303849501804506]],

    [[[0.24313095243684713, 1.47816562591576, 3.031111919828081],
      [0.09390330715359192, -0.06827601284567286, 0.5867636771002341, 0.8013918780542562]],
      [0,0,0],
      [1.44482955659220, -0.216311161075211, -0.543202748289152, -0.350785350870814, 0.562142814999080, 0.330500833933308]],

    [[[0.25012585110216445, 1.4887231619549708, 3.022576229428772],
      [0.09404116295393089, -0.06430768349697756, 0.5809189506519857, 0.8059491015485534]],
      [0,0,0],
      [1.44395436934484, -0.212121130812892, -0.540643208468406, -0.376688104981198, 0.566214745897564, 0.357087218123030]],

    [[[0.2751429619513606, 1.5201996536067093, 2.9953738537899186],
      [0.09346153220568144, -0.05361735809437773, 0.5639858198518068, 0.8187246887170503]],
      [0,0,0],
      [1.43757104118872, -0.199652287275526, -0.531730382076683, -0.442720215137760, 0.576671205238273, 0.425447541152677]],

    [[[0.3005781301294082, 1.5504426529475597, 2.9679176046390503],
      [0.09348863643853492, -0.04409224071221619, 0.54647878907822, 0.8310696013291797]],
      [0,0,0],
      [1.43104725676384, -0.187182804303549, -0.522635272537033, -0.508192235626281, 0.587145075203377, 0.493386092664958]],

    [[[0.3640694399736225, 1.616529409300072, 2.9018641945011083],
      [0.09453169331724073, -0.026429485538928547, 0.504741266851892, 0.8576721371185321]],
      [0,0,0],
      [1.41288420829881, -0.158172188488543, -0.500023731851246, -0.651679499001972, 0.610332795160824, 0.643223223540908]],

    [[[0.4291872634437962, 1.6759714084116817, 2.834747843823749],
      [0.09811876791733182, -0.01524163079812759, 0.4616973743498493, 0.8814623841046499]],
      [0,0,0],
      [1.39401760659016, -0.129183170743274, -0.476482429767977, -0.793002014759262, 0.633333493998426, 0.791531931702722]],

    [[[0.514937206951067, 1.7430818485607447, 2.747072368774042],
      [0.10552981893456144, -0.010481594623447584, 0.40661275071405256, 0.9074247431310986]],
      [0,0,0],
      [1.36824160679729, -0.0919738948239638, -0.444757999691220, -0.970675079389668, 0.662209735465884, 0.979049791486034]],

    [[[0.6010974626083683, 1.8003794000141018, 2.658558332618831],
      [0.11557872189464558, -0.016461625887732652, 0.35413475565302305, 0.9278788437920686]],
      [0,0,0],
      [1.34134740082980, -0.0548243143282690, -0.411647637430888, -1.14630585708680, 0.690519643279893, 1.16537560265068]],

    [[[0.7448570153535916, 1.8506802341282445, 2.5309573471112343],
      [0.11143869937497208, -0.019707037128781957, 0.3003537013191447, 0.9470906519827494]],
      [0,0,0],
      [1.28323694125015, -0.0111156279771829, -0.364081582309403, -1.28089214804311, 0.710599814388739, 1.31350100594634]],

    [[[0.8905044951571283, 1.8807686322298025, 2.399650754410078],
      [0.10736140924302028, -0.028629315394313085, 0.25110268880889636, 0.9615619219677681]],
      [0,0,0],
      [1.22466030081066, 0.0323604205607017, -0.315553738953506, -1.41515423979013, 0.729956211579435, 1.46158958900222]],

    [[[1.0355010910640943, 1.8912497103133417, 2.2652719658105447],
      [0.1029105005182895, -0.04202674506020687, 0.20786906336616964, 0.9718197539039793]],
      [0,0,0],
      [1.16563617424519, 0.0756196822116754, -0.266206347640800, -1.54938837198793, 0.748695672183763, 1.60983737877064]],

    [[[1.1774711955077146, 1.8829584996813478, 2.1283475139678023],
      [0.09782807878435817, -0.05844305097603802, 0.17184943232578734, 0.9785100149734178]],
      [0,0,0],
      [1.10617309070486, 0.118676719027275, -0.216191170184484, -1.68382347109941, 0.766939641993791, 1.75835954865157]],

    [[[1.3454360448933917, 1.845122010974218, 1.9604464983688827],
      [0.08784681511598974, -0.07469883717404734, 0.13761045169056485, 0.9837511801183981]],
      [0,0,0],
      [1.03160273578671, 0.168843415079182, -0.155749357352135, -1.82877783065120, 0.785492051298404, 1.91965826366872]],

    [[[1.5023614314327767, 1.7822937686003217, 1.7902916028330855],
      [0.0774133480030068, -0.08985645346560839, 0.11342945536078874, 0.9864414579578978]],
      [0,0,0],
      [0.956580382682594, 0.218753423455632, -0.0948572242585361, -1.97446728362232, 0.803672180463389, 2.08141260193357]],

    [[[1.644749420253658, 1.696764585325149, 1.6187534842724645],
      [0.06720106071413318, -0.10159299771252035, 0.09930018183496857, 0.9875739740101606]],
      [0,0,0],
      [0.881081936244090, 0.268423769421690, -0.0337361934442918, -2.12101205761270, 0.821616304060451, 2.24353790729741]],

    [[[1.769438779325662, 1.5911366638507913, 1.4467410342716613],
      [0.05818596854343251, -0.10773687035518711, 0.09463021133426568, 0.987953583390214]],
      [0,0,0],
      [0.805076593295838, 0.317867563222893, 0.0273942072286103, -2.26849635336742, 0.839419356166173, 2.40589514556731]],

    [[[1.8891135229539704, 1.439166074859765, 1.2814809254461386],
      [0.03843123330812124, -0.08562628978973727, 0.07028795588669519, 0.9931015970486257]],
      [0,0,0],
      [0.720428154307021, 0.358512472740708, 0.0848233533847481, -2.32496457704804, 0.841679978945838, 2.47441143181705]],

    [[[1.9837360863845581, 1.2703591023377019, 1.1195887565578806],
      [0.021927203057556258, -0.060085738719456196, 0.04646816169582113, 0.9968699071178851]],
      [0,0,0],
      [0.635823816408264, 0.398953026502538, 0.142351693930047, -2.38191129830580, 0.843790474472692, 2.54317465798279]],

    [[[2.051163415782638, 1.0891138118310364, 0.9623675676596019],
      [0.009116488729790118, -0.031319003069723426, 0.023012936121257706, 0.9992028895329702]],
      [0,0,0],
      [0.551238508956564, 0.439185322792744, 0.199952712783210, -2.43933483647210, 0.845754152120690, 2.61217023334455]],

    [[[2.0900424627375256, 0.900087325289494, 0.8110843102321388],
      [0.0004138321191437721, 0.0003945940813239295, -0.00023736172698716386, 0.999999808348931]],
      [0,0,0],
      [0.466646839005564, 0.479201827385092, 0.257602704566028, -2.49723673415436, 0.847571258254224, 2.68138196572325]]
]

if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 1

    test_code(test_cases[1])
    # test_code(test_cases[1])
    # test_code(test_cases[2])
    # test_code(test_cases[3])

    for i in range(0,len(extra_test_cases)):
        err = test_code(extra_test_cases[i])
        if err > 0.2:
            print("--------------------------------- ", i, err)
