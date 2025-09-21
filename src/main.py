import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define Functions
t, g, m1, m2, m3, l1, l2, l3 = smp.symbols("t, g, m_1, m_2, m_3, l_1, l_2, l_3")
th1, th2, th3 = smp.symbols(r"\theta_1, \theta_2, \theta_3", cls=smp.Function)

th1 = th1(t)
thdot1 = smp.diff(th1, t)
thddot1 = smp.diff(thdot1, t)

th2 = th2(t)
thdot2 = smp.diff(th2, t)
thddot2 = smp.diff(thdot2, t)

th3 = th3(t)
thdot3 = smp.diff(th3, t)
thddot3 = smp.diff(thdot3, t)

x1, y1, x2, y2, x3, y3 = smp.symbols("x_1, y_1, x_2, y_2, x_3, y_3", cls=smp.Function)

x1 = x1(l1, th1)
y1 = y1(l1, th1)
x2 = x2(l1, th1, l2, th2)
y2 = y2(l1, th1, l2, th2)
x3 = x3(l1, th1, l2, th2, l3, th3)
y3 = y3(l1, th1, l2, th2, l3, th3)

x1 = l1*smp.sin(th1)
y1 = -l1*smp.cos(th1)
x2 = l1*smp.sin(th1) + l2*smp.sin(th2)
y2 = -l1*smp.cos(th1) - l2*smp.cos(th2)
x3 = l1*smp.sin(th1) + l2*smp.sin(th2) + l3*smp.sin(th3)
y3 = -l1*smp.cos(th1) - l2*smp.cos(th2) - l3*smp.cos(th3)

# Define Energies
K1 = smp.Rational(1/2) * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
K2 = smp.Rational(1/2) * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
K3 = smp.Rational(1/2) * m3 * (smp.diff(x3, t)**2 + smp.diff(y3, t)**2)
U1 = m1 * g * y1
U2 = m2 * g * y2
U3 = m3 * g * y3

# Define Lagrangians
L1 = K1 - U1
L2 = K2 - U2
L3 = K3 - U3
L = L1 + L2 + L3

# Compute Euler-Lagrange Equations
EL1 = smp.diff(L, th1) - smp.diff(smp.diff(L, thdot1), t)
EL2 = smp.diff(L, th2) - smp.diff(smp.diff(L, thdot2), t)
EL3 = smp.diff(L, th3) - smp.diff(smp.diff(L, thdot3), t)
EL1 = EL1.simplify()
EL2 = EL2.simplify()
EL3 = EL3.simplify()

# Isolate Second Derivatives
EQ = smp.solve([EL1, EL2, EL3], (thddot1, thddot2, thddot3))

# Create Numerical Functions
thddot1_f = smp.lambdify((g, m1, m2, m3, l1, l2, l3, th1, th2, th3, thdot1, thdot2, thdot3), EQ[thddot1])
thdot1_f = smp.lambdify(thdot1, thdot1)
thddot2_f = smp.lambdify((g, m1, m2, m3, l1, l2, l3, th1, th2, th3, thdot1, thdot2, thdot3), EQ[thddot2])
thdot2_f = smp.lambdify(thdot2, thdot2)
thddot3_f = smp.lambdify((g, m1, m2, m3, l1, l2, l3, th1, th2, th3, thdot1, thdot2, thdot3), EQ[thddot3])
thdot3_f = smp.lambdify(thdot3, thdot3)

x1_f = smp.lambdify((l1, th1), x1)
y1_f = smp.lambdify((l1, th1), y1)
x2_f = smp.lambdify((l1, th1, l2, th2), x2)
y2_f = smp.lambdify((l1, th1, l2, th2), y2)
x3_f = smp.lambdify((l1, th1, l2, th2, l3, th3), x3)
y3_f = smp.lambdify((l1, th1, l2, th2, l3, th3), y3)

# Coupled Differential Equations
def dYdt(Y, t, g, m1, m2, m3, l1, l2, l3):
    th1, thdot1, th2, thdot2, th3, thdot3 = Y
    return [
        thdot1_f(thdot1),
        thddot1_f(g, m1, m2, m3, l1, l2, l3, th1, th2, th3, thdot1, thdot2, thdot3),
        thdot2_f(thdot2),
        thddot2_f(g, m1, m2, m3, l1, l2, l3, th1, th2, th3, thdot1, thdot2, thdot3),
        thdot3_f(thdot3),
        thddot3_f(g, m1, m2, m3, l1, l2, l3, th1, th2, th3, thdot1, thdot2, thdot3),
        ]

# Define Constants and Initial Values
t = np.arange(0, 10, 0.01)
g = 9.81
m1 = 1
m2 = 1
m3 = 1
l1 = 1
l2 = 1
l3 = 1

th01 = np.radians(90)
th02 = np.radians(90)
th03 = np.radians(90)
thdot01 = 0
thdot02 = 0
thdot03 = 0

# Solve Differential Equation Over the Time Interval t
Y0 = [th01, thdot01, th02, thdot02, th03, thdot03]
Y = odeint(dYdt, Y0, t, args=(g, m1, m2, m3, l1, l2, l3,))

# Transpose Array
th1_sol, th2_sol, th3_sol = Y.T[::2]

# Plot Path of System
plt.plot(x1_f(l1, th1_sol), y1_f(l1, th1_sol))
plt.plot(x2_f(l1, th1_sol, l2, th2_sol), y2_f(l1, th1_sol, l2, th2_sol))
plt.plot(x3_f(l1, th1_sol, l2, th2_sol, l3, th3_sol), y3_f(l1, th1_sol, l2, th2_sol, l3, th3_sol))
