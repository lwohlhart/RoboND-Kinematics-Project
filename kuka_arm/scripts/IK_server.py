#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
import numpy as np
from sympy import sqrt, atan2, acos, symbols, pi, cos, sin, simplify, sympify
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
T0_G = T0_6 * T6_G

R0_3 = T0_1[0:3,0:3]*T1_2[0:3,0:3]*T2_3[0:3,0:3]

def Rot_X(angle):
    return rot_axis1(angle).transpose()

def Rot_Y(angle):
    return rot_axis2(angle).transpose()

def Rot_Z(angle):
    return rot_axis3(angle).transpose()
            
R_corr = simplify(Rot_Z(pi) * Rot_Y(-pi/2))

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

    def ee_error(self, ee):
        fk_ee = T0_G.evalf(subs={q1:self.theta1, q2:self.theta2, q3:self.theta3, q4:self.theta4, q5:self.theta5, q6:self.theta6})[:3,3]
        ee_x_e = abs(fk_ee[0]-ee[0])
        ee_y_e = abs(fk_ee[1]-ee[1])
        ee_z_e = abs(fk_ee[2]-ee[2])
        return sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)

    def orientation_error(self, orientation):        
        corr = Matrix.eye(4)
        corr[:3,:3] = R_corr
        rot_mat = T0_G.evalf(subs={q1:self.theta1, q2:self.theta2, q3:self.theta3, q4:self.theta4, q5:self.theta5, q6:self.theta6})
        rot_mat = rot_mat * corr
        fk_orientation = tf.transformations.quaternion_from_matrix(rot_mat.tolist())
        err = np.linalg.norm(fk_orientation - np.array([orientation.x, orientation.y, orientation.z, orientation.w]))
        return err

    def fk_error(self, wc, ee, orientation):

        return self.wc_error(wc) + self.ee_error(ee) + self.orientation_error(orientation)


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

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        ### Your FK code here
        # Create symbols
        #
        #
        # Create Modified DH parameters
        #
        #
        # Define Modified DH Transformation matrix
        #
        #
        # Create individual transformation matrices
        #
        #
        # Extract rotation matrices from the transformation matrices
        #
        #
        ###

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
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
            joints.append(JointsOption())
            joints[0].theta1 = atan2(wy, wx)
            joints[1].theta1 = atan2(-wy,-wx) # 2nd_alternative    

            # Calculate theta2 and theta3 from triangle spanned by joint2, joint3 and wrist-center
            joints[0].wx_tilde = sqrt(wx * wx + wy * wy) - dh_params[a1] # 1st_alternative
            joints[0].wz_tilde = wz - dh_params[d1]
            joints[1].wx_tilde = -sqrt(wx * wx + wy * wy) - dh_params[a1] # 2nd_alternative
            joints[1].wz_tilde = wz - dh_params[d1]

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
                R3_6 = R0_3_current.inv("LU") * Rrpy

                # R3_6(q4, q5, q6) structure:
                #[[-sin(q4)*sin(q6) + cos(q4)*cos(q5)*cos(q6), -sin(q4)*cos(q6) - sin(q6)*cos(q4)*cos(q5), -sin(q5)*cos(q4)],
                # [                           sin(q5)*cos(q6),                           -sin(q5)*sin(q6),          cos(q5)],
                # [-sin(q4)*cos(q5)*cos(q6) - sin(q6)*cos(q4),  sin(q4)*sin(q6)*cos(q5) - cos(q4)*cos(q6),  sin(q4)*sin(q5)]]
                
                j.theta4 = atan2(R3_6[2,2], -R3_6[0,2])
                j.theta5 = atan2(sqrt(R3_6[0,2] * R3_6[0,2] + R3_6[2,2] * R3_6[2,2] ), R3_6[2,2])
                j.theta6 = atan2(-R3_6[1,1], R3_6[1,0])

                j_alt = j.clone()
                j_alt.theta5 = (acos(R3_6[1,2])) # ambiguous but numerically more accurate alternative
                joints_alt.append(j_alt)

                # j_alt = j.clone()
                # j_alt.theta6 = atan2(R3_6[1,1], -R3_6[1,0])
                # joints_alt.append(j_alt)

                # j_alt = j.clone()        
                # j_alt.theta4 = atan2(-R3_6[2,2], R3_6[0,2])
                # joints_alt.append(j_alt)

                # j_alt = j.clone()        
                # j_alt.theta4 = atan2(-R3_6[2,2], R3_6[0,2])
                # j_alt.theta6 = atan2(R3_6[1,1], -R3_6[1,0])
                # joints_alt.append(j_alt)        


            joints = [j for j in (joints + joints_alt) if j.check_valid()]
            # for j in joints:
            #     j.debug_print()
            #     print(j.fk_error([wx, wy, wz], [px, py, pz], req.poses[x].orientation))

            # print(len(joints))
            joints = sorted(joints, key = lambda j: j.fk_error([wx, wy, wz], [px, py, pz], req.poses[x].orientation))
            (theta1, theta2, theta3, theta4, theta5, theta6) = joints[0].thetas()

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
