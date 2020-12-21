from simulation_functions import *
import numpy as np
import scipy.integrate as spint


# Context

dt = 0.0001  # Période d'échantillonnage des signaux
g = 9.81  # Accélération de la pesanteur
mA = 1  # Masse de la caméra et du support
d = 3.2
s = 2.2
v = 2.0
STADE = [d, s, v]  # Dimension du stade
R = 0.15  # Rayon des treuils
R1 = 0.917
R2 = 0.917
R3 = 0.917  # Résistance d'induit des moteurs
J1 = 1.5859
J2 = 1.5859
J3 = 1.5859  # Moment d'inertie des treuils
kI1 = 2.51945
kI2 = 2.51945
kI3 = 2.51945  # Constante liant couple et courant des moteurs
kE1 = 3.3942
kE2 = 3.3942
kE3 = 3.3942  # Constante liant vitesse et fem des moteurs
la1 = 0.0670
la2 = 0.0670
la3 = 0.0670  # Coefficient de frottement des moteurs
Kpos = 4200
Kvit = 130  # Gains des correcteurs
G = np.diag([R1 * J1 / kI1, R2 * J2 / kI2, R3 * J3 / kI3])  # Construction des matrices G, L et S
L = np.diag([kE1 + la1 * R1 / kI1, kE2 + la2 * R2 / kI2, kE3 + la3 * R3 / kI3])
S = np.diag([R1 / kI1, R2 / kI2, R3 / kI3])
Sinv = np.linalg.inv(S)


def Force_pert(t):
    """
        Force de perturbation
    """
    Fp = np.array([[(100 * (sin(4 * pi * t) + sin(32 * pi * t)))], [0], [0]])
    return Fp


tfin = 2.8
t1 = 0.6
t2 = 2.25
Xdep = np.array([[0.3], [1.8], [-0.2]])
Xfin = np.array([[0.9], [1.2], [-0.9]])

time, vitXref, posXref, vitThref, posThref = TrajRef(Xdep, Xfin, tfin, t1, t2, STADE, R, timestep=dt)
t = 1.5
X_pt0, X0, Th_pt0, Th0 = CalcConsigne(t, time, vitXref, posXref, vitThref, posThref)
print("Vmax=", np.sqrt(np.sum(X_pt0 * X_pt0)))

# Initial conditions
Y0 = np.zeros(12)
Y0[0:3] = Xdep.transpose();

# Resolution
Ysol = spint.odeint(EqDiff, Y0, time, args=(
    STADE, R, mA, Sinv, L, G, g, Kpos, Kvit, time, vitXref, posXref, vitThref, posThref, Force_pert))

x = Ysol[:, 0]
y = Ysol[:, 1]
z = Ysol[:, 2]
xpt = Ysol[:, 3]
ypt = Ysol[:, 4]
zpt = Ysol[:, 5]
TH1 = Ysol[:, 6]
TH2 = Ysol[:, 7]
TH3 = Ysol[:, 8]
TH1pt = Ysol[:, 9]
TH2pt = Ysol[:, 10]
TH3pt = Ysol[:, 11]

# Tension forces
xptpt = np.gradient(xpt, dt)
yptpt = np.gradient(ypt, dt)
zptpt = np.gradient(zpt, dt)
F1 = np.zeros(len(time))
F2 = np.zeros(len(time))
F3 = np.zeros(len(time))
for k in range(len(time)):
    X = np.array([[x[k]], [y[k]], [z[k]]])
    Xptpt = np.array([[xptpt[k]], [yptpt[k]], [zptpt[k]]])
    P = CalcPXTH(X, STADE, R)
    F = mA / R * np.dot(np.linalg.inv(np.transpose(P)), Xptpt + np.array([[0], [0], [g]]) - Force_pert(time[k]) / mA)
    F1[k] = F[0]
    F2[k] = F[1]
    F3[k] = F[2]

# Tension forces upon motors
U1 = np.zeros(len(time))
U2 = np.zeros(len(time))
U3 = np.zeros(len(time))
for k in range(len(time)):
    X_pt0, X0, Th_pt0, Th0 = CalcConsigne(time[k], time, vitXref, posXref, vitThref, posThref)
    U1[k] = Kpos * (Th0[0] - TH1[k]) + Kvit * (Th_pt0[0] - TH1pt[k])
    U2[k] = Kpos * (Th0[1] - TH2[k]) + Kvit * (Th_pt0[1] - TH2pt[k])
    U3[k] = Kpos * (Th0[2] - TH3[k]) + Kvit * (Th_pt0[2] - TH3pt[k])