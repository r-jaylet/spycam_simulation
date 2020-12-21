from numpy import *

'CONTEXT'
D = 0.75  # distance séparant les deux poulies [m]
H = 0.8  # hauteur des deux poulies [m]
m = 0.18  # masse de la caméra [kg]
g = 9.81  # acc de la pesanteur [m/s^2]
r = 0.028  # rayon de poulie

A1 = [0, H]
A2 = [D, H]

L10 = ((A1[0]) ** 2 + (A1[1]) ** 2) ** (1 / 2)
L20 = ((A2[0]) ** 2 + (A2[1]) ** 2) ** (1 / 2)

T = arange(0, 2 * pi, 0.1)

'MODELE GEOMETRIQUE DIRECT'


def th1(M):
    return -arctan(((A1[1] - M[1]) / (A1[0] - M[0])))


def th2(M):
    return arctan((A2[1] - M[1]) / (A2[0] - M[0]))


def lo1(M):
    return ((M[0] - A1[0]) ** 2 + (M[1] - A1[1]) ** 2) ** (1 / 2)


def lo2(M):
    return ((M[0] - A2[0]) ** 2 + (M[1] - A1[1]) ** 2) ** (1 / 2)


def tens(M):
    return [(m * g * cos(th2(M))) / sin(th1(M) + th2(M)), (m * g * cos(th1(M))) / sin(th1(M) + th2(M))]


'definition trajectoire circulaire'
C = [0.25, 0.25]
R = 0.10

X = C[0] + R * cos(T)
Y = C[1] + R * sin(T)
M = [X, Y]

VX = -R * sin(T)
VY = R * cos(T)
AX = -R * cos(T)
AY = -R * sin(T)

'EVOLUTION DES ANGLES'


def thetadegre(M):
    t1 = []
    t2 = []
    for i in range(len(T)):
        t1.append(-180 * (arctan((A1[1] - M[1][i]) / (A1[0] - M[0][i]))) / 3.14)
        t2.append(180 * (arctan((A2[1] - M[1][i]) / (A2[0] - M[0][i]))) / 3.14)
    return [t1, t2]


thetaz1 = (thetadegre(M))[0]
thetaz2 = (thetadegre(M))[1]


def theta(M):
    t1 = []
    t2 = []
    for i in range(len(T)):
        t1.append(-arctan((A1[1] - M[1][i]) / (A1[0] - M[0][i])))  # attention signe pour derivées
        t2.append(arctan((A2[1] - M[1][i]) / (A2[0] - M[0][i])))
    return [t1, t2]


theta1 = (theta(M))[0]
theta2 = (theta(M))[1]

'EVOLUTION DES GRANDEURS CARACtERISTIQUES'


def longueur(M):
    L1 = []
    L2 = []
    for i in range(len(T)):
        L1.append(((M[0][i] - A1[0]) ** 2 + (M[1][i] - A1[1]) ** 2) ** (1 / 2))
        L2.append(((M[0][i] - A2[0]) ** 2 + (M[1][i] - A1[1]) ** 2) ** (1 / 2))
    return [L1, L2]


L1 = (longueur(M))[0]
L2 = (longueur(M))[1]


def thetad(M):
    t1 = []
    t2 = []
    for i in range(len(T)):
        t1.append((((-sin(theta1[i])) * VX[i]) / L1[i]) + (((cos(theta1[i])) * VY[i]) / L2[i]))
        t2.append((((-sin(theta2[i])) * VX[i]) / L2[i]) + (((cos(theta2[i])) * VY[i]) / L2[i]))
    return [t1, t2]


thetad1 = (thetad(M))[0]
thetad2 = (thetad(M))[1]


def longueurd(M):
    L1 = []
    L2 = []
    for i in range(len(T)):
        L1.append(cos(theta1[i]) * VX[i] + sin(theta1[i]) * VY[i])
        L2.append(cos(theta2[i]) * VX[i] + sin(theta2[i]) * VY[i])
    return [L1, L2]


Ld1 = (longueurd(M))[0]
Ld2 = (longueurd(M))[1]

'EVOLUTION TENSION CABLE'


def tension():
    F1 = []
    F2 = []
    for i in range(len(T)):
        F1.append((-m * AX[i] * sin(theta2[i]) + m * g * cos(theta2[i])) / sin(theta1[i] + theta2[i]))
        F2.append((m * AX[i] * sin(theta1[i]) + m * (g + AX[i]) * cos(theta1[i])) / sin(theta1[i] + theta2[i]))
    return [F1, F2]


Tension1 = tension()[0]
Tension2 = tension()[1]


def plan(seuil):
    i = []
    j = []
    for x in arange(0, D, 0.003):
        for y in arange(0, H, 0.003):
            M = [x, y]
            if (sin(th1(M) + th2(M))) != 0:
                if (m * g * (cos(th2(M)) / sin(th1(M) + th2(M)))) > seuil or (
                        m * g * (cos(th1(M)) / sin(th1(M) + th2(M)))) > seuil:
                    i.append(M[0])
                    j.append(M[1])
    k = 100 * (H - min(j)) / H
    return [i, j, k]


X = plan(5)[0]
Y = plan(5)[1]

'EVOLUTION ROTATION DES MOTEURS'


def beta(M):
    b1 = []
    b2 = []
    for i in range(len(T)):
        b1.append((L1[0] - L1[i]) / r)
        b2.append((L2[0] - L2[i]) / r)
    return [b1, b2]


beta1 = (beta(M))[0]
beta2 = (beta(M))[1]


def betad(M):
    bd1 = []
    bd2 = []
    for i in range(len(T)):
        bd1.append(-(((M[0][i] - A1[0]) * VX[i]) + ((M[1][i] - A1[1]) * VY[i])) / (r * L1[i]))
        bd2.append((((M[0][i] - A2[0]) * VX[i]) + ((M[1][i] - A2[1]) * VY[i])) / (r * L2[i]))
    return [bd1, bd2]


betad1 = (betad(M))[0]
betad2 = (betad(M))[1]


def betadd(M):
    bdd1 = []
    bdd2 = []
    for i in range(len(T)):
        bdd1.append(((-(((VX[i]) ** 2 + (M[0][i] - A1[0]) * AX[i]) + (VY[i]) ** 2 + (M[1][i] - A1[1]) * AY[i]) * Ld1[
            i]) / (r * (L1[i]) ** 2)) + betad1[i])
        bdd2.append(((-(((VX[i]) ** 2 + (M[1][i] - A2[0]) * AX[i]) + (VY[i]) ** 2 + (M[1][i] - A2[1]) * AY[i]) * Ld2[
            i]) / (r * (L2[i]) ** 2)) + betad2[i])
    return [bdd1, bdd2]


betadd1 = (betadd(M))[0]
betadd2 = (betadd(M))[1]


def multi(L, k):
    A = []
    for i in range(len(L)):
        A.append((k * L[i]))
    return A


vitesseRPM1 = multi(betad1, (60 / (2 * 3.14)))
vitesseRPM2 = multi(betad2, (60 / (2 * 3.14)))

'COMMANDE ARDUINO'


def adaptation(vRPM1, vRPM2):
    L1 = []
    L2 = []
    for i in range(len(vRPM1)):
        if vRPM1[i] <= 0:
            L1.append(int(5.84 * vRPM1[i] + 1529))
        elif vRPM1[i] > 0:
            L1.append(int(5.95 * vRPM1[i] + 1487))
        else:
            L1.append(0)
        if vRPM2[i] <= 0:
            L2.append(int(5.32 * vRPM2[i] + 1456.4))
        elif vRPM2[i] > 0:
            L2.append(int(5.56 * vRPM2[i] + 1499.1))
        else:
            L2.append(0)
    return L1, L2


vitessearduino1 = adaptation(vitesseRPM1, vitesseRPM2)[0]
vitessearduino2 = adaptation(vitesseRPM1, vitesseRPM2)[1]


def arduino(L1, L2):
    for i in range(len(L1)):
        print("servo1.writeMicroseconds(" + str(L1[i]) + ");")
        print("servo2.writeMicroseconds(" + str(L2[i]) + "); delay(200);")
