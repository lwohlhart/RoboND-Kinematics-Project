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
from sympy import *
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

T0_2 = simplify(T0_1 * T1_2)
T0_3 = simplify(T0_2 * T2_3)
T0_4 = simplify(T0_3 * T3_4)
T0_5 = simplify(T0_4 * T4_5)
T0_6 = simplify(T0_5 * T5_6)
T0_G = simplify(T0_6 * T6_G)

def Rot_X(angle):
    return rot_axis1(angle).transpose()

def Rot_Y(angle):
    return rot_axis2(angle).transpose()

def Rot_Z(angle):
    return rot_axis3(angle).transpose()
            
R_corr = simplify(Rot_Z(pi) * Rot_Y(-pi/2))

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
            wx, wy, wz = Matrix([px, py, pz]) - (dh_params.d6 + dh_params.d7) * Rrpy[:,2]

            # Calculate joint angles using Geometric IK method
            #
            #
            ###
            theta1 = atan2(wy, wx)
            
            # Calculate theta2 and theta3 from triangle spanned by joint2, joint3 and wrist-center
            tri_fwd = sqrt(wx * wx + wy * wy) - dh_params.a1
            tri_up = wz - dh_params.d1

            len_A = sqrt(dh_params.d4 * dh_params.d4 + dh_params.a3 * dh_params.a3)
            len_B = sqrt(tri_fwd * tri_fwd + tri_up * tri_up)
            len_C = dh_params.a2

            theta2 = pi/2 - acos((len_B*len_B + len_C*len_C - len_A*len_A)/(2*len_B*len_C)) - atan2(tri_up, tri_fwd)
            theta3 = acos((len_A*len_A + len_C*len_C - len_B*len_B)/(2*len_A*len_C)) - pi/2

            R0_3 = T0_3[0:3,0:3]
            R0_3 = R0_3.subs({q1:theta1, q2:theta2, q3: theta3})
            R3_6 = R0_3.inv("LU") * Rrpy

            # attempt 1
            theta4 = atan2(R3_6[1,2], R3_6[0,2])
            theta5 = atan2(sqrt(R3_6[0,2] * R3_6[0,2] + R3_6[1,2] * R3_6[1,2] ), R3_6[2,2])
            theta6 = atan2(R3_6[2,1], -R3_6[2,0])

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
