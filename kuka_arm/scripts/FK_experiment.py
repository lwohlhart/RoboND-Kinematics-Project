import numpy as np

from sympy import symbols, cos, sin, pi, simplify, sqrt, atan2
from sympy.matrices import Matrix

q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

s = {alpha0:     0, a0:      0, d1:  0.75,
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

T0_1 = dh_transform(alpha0, a0, q1, d1).subs(s)
T1_2 = dh_transform(alpha1, a1, q2, d2).subs(s)
T2_3 = dh_transform(alpha2, a2, q3, d3).subs(s)
T3_4 = dh_transform(alpha3, a3, q4, d4).subs(s)
T4_5 = dh_transform(alpha4, a4, q5, d5).subs(s)
T5_6 = dh_transform(alpha5, a5, q6, d6).subs(s)
T6_G = dh_transform(alpha6, a6, q7, d7).subs(s)

T0_2 = simplify(T0_1 * T1_2)
T0_3 = simplify(T0_2 * T2_3)
T0_4 = simplify(T0_3 * T3_4)
T0_5 = simplify(T0_4 * T4_5)
T0_6 = simplify(T0_5 * T5_6)
T0_G = simplify(T0_6 * T6_G)


print("T0_1: ", T0_1.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))
print("T0_2: ", T0_2.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))
print("T0_3: ", T0_3.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))
print("T0_4: ", T0_4.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))
print("T0_5: ", T0_5.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))
print("T0_6: ", T0_6.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))
print("T0_G: ", T0_G.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))

R_z = Matrix([[cos(pi), -sin(pi), 0, 0],
              [sin(pi),  cos(pi), 0, 0],
              [      0,        0, 1, 0],
              [      0,        0, 0, 1]])

R_y = Matrix([[ cos(-pi/2), 0, sin(-pi/2), 0],
              [          0, 1,          0, 0],
              [-sin(-pi/2), 0, cos(-pi/2), 0],
              [          0, 0,          0, 1]])              
R_corr = simplify(R_z * R_y)

T_gripper = simplify(T0_G * R_corr)

print("T_gripper: ", T_gripper.evalf(subs={q1:0, q2:0, q3:0, q4:0 , q5:0 , q6:0}))
