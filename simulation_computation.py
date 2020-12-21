from IPython import get_ipython

get_ipython().magic('clear')
get_ipython().magic('reset -sf')

from math import *
from simulation_functions import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint

"CONTEXTE"

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

"Résolution de l'équation différentielle"

# Conditions intiales
Y0 = np.zeros(12)
Y0[0:3] = Xdep.transpose();

# résolution
Ysol = spint.odeint(EqDiff, Y0, time, args=(
STADE, R, mA, Sinv, L, G, g, Kpos, Kvit, time, vitXref, posXref, vitThref, posThref, Force_pert))

# Extraction des valeurs à partir des solutions
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

# Calcul des forces de tension sur les 3 treuils
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

# Calcul de la tension aux bornes des moteurs
U1 = np.zeros(len(time))
U2 = np.zeros(len(time))
U3 = np.zeros(len(time))
for k in range(len(time)):
    X_pt0, X0, Th_pt0, Th0 = CalcConsigne(time[k], time, vitXref, posXref, vitThref, posThref)
    U1[k] = Kpos * (Th0[0] - TH1[k]) + Kvit * (Th_pt0[0] - TH1pt[k])
    U2[k] = Kpos * (Th0[1] - TH2[k]) + Kvit * (Th_pt0[1] - TH2pt[k])
    U3[k] = Kpos * (Th0[2] - TH3[k]) + Kvit * (Th_pt0[2] - TH3pt[k])

"Tracé des résultats"

# Vitesse de référence de la caméra
plt.figure(0)
plt.plot(time, vitXref[0, :], 'r--')
plt.plot(time, vitXref[1, :], 'g--')
plt.plot(time, vitXref[2, :], 'b--')
plt.title("Vitesse de référence")
plt.xlabel("temps (s)");
plt.ylabel("vitesse (m/s)")
plt.legend(["$\\dot{x}^0$", "$\\dot{y}^0$", "$\\dot{z}^0$"])
plt.show()

# Position de la caméra
plt.figure(1)
plt.plot(time, posXref[0, :], 'r--')
plt.plot(time, x, 'r')
plt.plot(time, posXref[1, :], 'g--')
plt.plot(time, y, 'g')
plt.plot(time, posXref[2, :], 'b--')
plt.plot(time, z, 'b')
plt.title("Position de la caméra")
plt.xlabel("temps (s)");
plt.ylabel("position (m)")
plt.legend(["${x}^0$", "${x}$", "${y}^0$", "${y}$", "${z}^0$", "${z}$"])
plt.show()

# -Erreur de position
plt.figure(2)
plt.plot(time, (x - posXref[0, :]) * 1e3, 'r')
plt.plot(time, (y - posXref[1, :]) * 1e3, 'g')
plt.plot(time, (z - posXref[2, :]) * 1e3, 'b')
plt.title("Erreur de position de la caméra")
plt.xlabel("temps (s)");
plt.ylabel("erreur (mm)")
plt.legend(["$x-{x}^0$", "$y-{y}^0$", "$z-{z}^0$"])
plt.show()

# Vitesse de rotation de référence des treuils
plt.figure(3)
plt.plot(time, vitThref[0, :], 'r--')
plt.plot(time, vitThref[1, :], 'g--')
plt.plot(time, vitThref[2, :], 'b--')
plt.title("Vitesse de rotation de référence des moteurs")
plt.xlabel("temps (s)");
plt.ylabel("vitesse (rad/s)")
plt.legend(["$\\dot{\\theta}_1^0$", "$\\dot{\\theta}_2^0$", "$\\dot{\\theta}_3^0$"])
plt.show()

# Position angulaire des treuils
plt.figure(4)
plt.plot(time, posThref[0, :], 'r--')
plt.plot(time, TH1, 'r')
plt.plot(time, posThref[1, :], 'g--')
plt.plot(time, TH2, 'g')
plt.plot(time, posThref[2, :], 'b--')
plt.plot(time, TH3, 'b')
plt.title("Position angulaire des moteurs")
plt.xlabel("temps (s)");
plt.ylabel("angle (rad)")
plt.legend(["${\\theta}_1^0$", "${\\theta}_1$", "${\\theta}_2^0$", "${\\theta}_2$", "${\\theta}_3^0$", "${\\theta}_3$"])
plt.show()

# Erreur angulaire de positionnement
plt.figure(5)
plt.plot(time, (TH1 - posThref[0, :]) * 1e3, 'r')
plt.plot(time, (TH2 - posThref[1, :]) * 1e3, 'g')
plt.plot(time, (TH3 - posThref[2, :]) * 1e3, 'b')
plt.title("Erreur de position angulaire des moteurs")
plt.xlabel("temps (s)");
plt.ylabel("erreur (mrad)")
plt.legend(["${\\theta}_1-{\\theta}_1^0$", "${\\theta}_2-{\\theta}_2^0$", "${\\theta}_3-{\\theta}_3^0$"])
plt.show()

# Force sur les treuils
plt.figure(6)
plt.plot(time, F1, 'r')
plt.plot(time, F2, 'g')
plt.plot(time, F3, 'b')
plt.title("Force de tension au niveau des 3 treuils")
plt.xlabel("temps (s)");
plt.ylabel("Force (N)")
plt.legend(["$F_1$", "$F_2$", "$F_3$"])
plt.show()

# Tension de commande des moteurs
plt.figure(7)
plt.plot(time, U1, 'r')
plt.plot(time, U2, 'g')
plt.plot(time, U3, 'b')
plt.title("Tension de commande des moteurs")
plt.xlabel("temps (s)");
plt.ylabel("Tension (V)")
plt.legend(["$U_1$", "$U_2$", "$U_3$"])
plt.show()
