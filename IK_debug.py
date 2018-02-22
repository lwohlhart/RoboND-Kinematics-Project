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
T4_5 = dh_transform(alpha4, a4, q5, d5).subs(dh_params)
T5_6 = dh_transform(alpha5, a5, q6, d6).subs(dh_params)
T6_G = dh_transform(alpha6, a6, q7, d7).subs(dh_params)

# T0_2 = simplify(T0_1 * T1_2)
# T0_3 = simplify(T0_2 * T2_3)
# T0_4 = simplify(T0_3 * T3_4)
# T0_5 = simplify(T0_4 * T4_5)
# T0_6 = simplify(T0_5 * T5_6)
# T0_G = simplify(T0_6 * T6_G)
T0_2 = T0_1 * T1_2
T0_3 = T0_2 * T2_3
T0_4 = T0_3 * T3_4
T0_5 = T0_4 * T4_5
T0_6 = T0_5 * T5_6
T0_G = simplify(T0_6 * T6_G)


# numpy version of the simplified T0_G
def np_T0_G(q1, q2, q3, q4, q5, q6):
    return np.array([
        [((np.sin(q1)*np.sin(q4) + np.sin(q2 + q3)*np.cos(q1)*np.cos(q4))*np.cos(q5) + np.sin(q5)*np.cos(q1)*np.cos(q2 + q3))*np.cos(q6) - (-np.sin(q1)*np.cos(q4) + np.sin(q4)*np.sin(q2 + q3)*np.cos(q1))*np.sin(q6), -((np.sin(q1)*np.sin(q4) + np.sin(q2 + q3)*np.cos(q1)*np.cos(q4))*np.cos(q5) + np.sin(q5)*np.cos(q1)*np.cos(q2 + q3))*np.sin(q6) + (np.sin(q1)*np.cos(q4) - np.sin(q4)*np.sin(q2 + q3)*np.cos(q1))*np.cos(q6), -(np.sin(q1)*np.sin(q4) + np.sin(q2 + q3)*np.cos(q1)*np.cos(q4))*np.sin(q5) + np.cos(q1)*np.cos(q5)*np.cos(q2 + q3), -0.303*np.sin(q1)*np.sin(q4)*np.sin(q5) + 1.25*np.sin(q2)*np.cos(q1) - 0.303*np.sin(q5)*np.sin(q2 + q3)*np.cos(q1)*np.cos(q4) - 0.054*np.sin(q2 + q3)*np.cos(q1) + 0.303*np.cos(q1)*np.cos(q5)*np.cos(q2 + q3) + 1.5*np.cos(q1)*np.cos(q2 + q3) + 0.35*np.cos(q1)],
        [ ((np.sin(q1)*np.sin(q2 + q3)*np.cos(q4) - np.sin(q4)*np.cos(q1))*np.cos(q5) + np.sin(q1)*np.sin(q5)*np.cos(q2 + q3))*np.cos(q6) - (np.sin(q1)*np.sin(q4)*np.sin(q2 + q3) + np.cos(q1)*np.cos(q4))*np.sin(q6), -((np.sin(q1)*np.sin(q2 + q3)*np.cos(q4) - np.sin(q4)*np.cos(q1))*np.cos(q5) + np.sin(q1)*np.sin(q5)*np.cos(q2 + q3))*np.sin(q6) - (np.sin(q1)*np.sin(q4)*np.sin(q2 + q3) + np.cos(q1)*np.cos(q4))*np.cos(q6), -(np.sin(q1)*np.sin(q2 + q3)*np.cos(q4) - np.sin(q4)*np.cos(q1))*np.sin(q5) + np.sin(q1)*np.cos(q5)*np.cos(q2 + q3),  1.25*np.sin(q1)*np.sin(q2) - 0.303*np.sin(q1)*np.sin(q5)*np.sin(q2 + q3)*np.cos(q4) - 0.054*np.sin(q1)*np.sin(q2 + q3) + 0.303*np.sin(q1)*np.cos(q5)*np.cos(q2 + q3) + 1.5*np.sin(q1)*np.cos(q2 + q3) + 0.35*np.sin(q1) + 0.303*np.sin(q4)*np.sin(q5)*np.cos(q1)],
        [                                                                -(np.sin(q5)*np.sin(q2 + q3) - np.cos(q4)*np.cos(q5)*np.cos(q2 + q3))*np.cos(q6) - np.sin(q4)*np.sin(q6)*np.cos(q2 + q3),                                                     (np.sin(q5)*np.sin(q2 + q3) - np.cos(q4)*np.cos(q5)*np.cos(q2 + q3))*np.sin(q6) - np.sin(q4)*np.cos(q6)*np.cos(q2 + q3),           -np.sin(q5)*np.cos(q4)*np.cos(q2 + q3) - np.sin(q2 + q3)*np.cos(q5),                                                                                 -0.303*np.sin(q5)*np.cos(q4)*np.cos(q2 + q3) - 0.303*np.sin(q2 + q3)*np.cos(q5) - 1.5*np.sin(q2 + q3) + 1.25*np.cos(q2) - 0.054*np.cos(q2 + q3) + 0.75],
        [                                                                                                                                                            0,                                                                                                                                                0,                                                              0,                                                                                                  1]])
def np_R0_3(q1, q2, q3):
    return np.array([[sin(q2 + q3)*cos(q1), cos(q1)*cos(q2 + q3), -sin(q1)],
                    [sin(q1)*sin(q2 + q3), sin(q1)*cos(q2 + q3),  cos(q1)],
                    [        cos(q2 + q3),        -sin(q2 + q3),        0]])

R0_3 = simplify(T0_1[0:3,0:3]*T1_2[0:3,0:3]*T2_3[0:3,0:3])

def Rot_X(angle):
    return rot_axis1(angle).transpose()

def Rot_Y(angle):
    return rot_axis2(angle).transpose()

def Rot_Z(angle):
    return rot_axis3(angle).transpose()

R_corr = simplify(Rot_Z(pi) * Rot_Y(-pi/2))

# simpler non parametric version
R_corr = Matrix([[0,  0, 1],
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
        if local_T0_G:
            return local_T0_G[:3,3]
        else:                
            return T0_G.evalf(subs={q1:self.theta1, q2:self.theta2, q3:self.theta3, q4:self.theta4, q5:self.theta5, q6:self.theta6})[:3,3]

    def ee_orientation(self, local_T0_G=None):
        corr = Matrix.eye(4)
        corr[:3,:3] = R_corr
        if local_T0_G:
            rot_mat = local_T0_G
        else:
            rot_mat = T0_G.evalf(subs={q1:self.theta1, q2:self.theta2, q3:self.theta3, q4:self.theta4, q5:self.theta5, q6:self.theta6})
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
        local_T0_G = T0_G.evalf(subs={q1:self.theta1, q2:self.theta2, q3:self.theta3, q4:self.theta4, q5:self.theta5, q6:self.theta6})            
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
        R0_3_current = R0_3.evalf(subs={q1:j.theta1, q2:j.theta2, q3: j.theta3})
        #R3_6 = R0_3_current.inv("LU") * Rrpy
        R3_6 = R0_3_current.T * Rrpy

        # R3_6(q4, q5, q6) structure:
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
    #     print(j.fk_error([wx, wy, wz], [px, py, pz], req.poses[x].orientation))

    # print(len(joints))
    joints = sorted(joints, key = lambda j: j.fk_error([wx, wy, wz], [px, py, pz], req.poses[x].orientation))
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
    your_ee = T0_G.evalf(subs={q1:theta1, q2: theta2, q3:theta3, q4:theta4, q5: theta5, q6: theta6})[:3,3]

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




if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 1

    test_code(test_cases[1])
    # test_code(test_cases[1])
    # test_code(test_cases[2])
    # test_code(test_cases[3])

    # for i in range(0,len(extra_test_cases)):
    #     err = test_code(extra_test_cases[i])
    #     if err > 0.2:
    #         print("--------------------------------- ", i, err)
