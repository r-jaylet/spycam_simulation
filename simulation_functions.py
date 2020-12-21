from math import *
import numpy as np
import scipy.integrate as spint


def CalcPXTH(X, STADE, R):
    """
    Calcul de la matrice P de changement de base 
    
    entrée :    X position (vecteur 3 composantes)
                STADE taille du stade (vecteur 3 composantes)   
                R rayon des treuils (scalaire)
    sortie : matrice 3x3
    """
    x = X[0]
    y = X[1]
    z = X[2]
    d = STADE[0]
    s = STADE[1]
    v = STADE[2]
    k = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    h = np.sqrt((d - x) ** 2 + y ** 2 + z ** 2)
    m = np.sqrt((d - x) ** 2 + (s - y) ** 2 + z ** 2)
    n = np.sqrt(x ** 2 + (s - y) ** 2 + z ** 2)

    P = np.zeros((3, 3))
    P[0, 0] = -(d - x) / m - (d - x) / h
    P[0, 1] = -(s - y) / m + y / h
    P[0, 2] = z / m + z / h
    P[1, 0] = -(d - x) / m + x / n
    P[1, 1] = -(s - y) / m - (s - y) / n
    P[1, 2] = z / m + z / n
    P[2, 0] = -(d - x) / m + x / n + x / k - (d - x) / h
    P[2, 1] = -(s - y) / m - (s - y) / n + y / k + y / h
    P[2, 2] = z / m + z / n + z / k + z / h

    return P / R


def CalcHXTH(X, STADE, R):
    """
    Calcul des matrices H1, H2, H3
    entrée :    X position (vecteur 3 composantes)
                STADE taille du stade (vecteur 3 composantes)   
                R rayon des treuils (scalaire)
    sortie : H1,H2,H3 matrices 3x3
    """
    x = X[0]
    y = X[1]
    z = X[2]
    d = STADE[0]
    s = STADE[1]
    v = STADE[0]
    k = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    h = np.sqrt((d - x) ** 2 + y ** 2 + z ** 2)
    m = np.sqrt((d - x) ** 2 + (s - y) ** 2 + z ** 2)
    n = np.sqrt(x ** 2 + (s - y) ** 2 + z ** 2)

    H1 = np.zeros((3, 3))
    H1[0, 0] = 1 / m + 1 / h - ((d - x) ** 2) / m ** 3 - ((d - x) ** 2) / h ** 3
    H1[0, 1] = -(d - x) * (s - y) / m ** 3 + (d - x) * y / h ** 3
    H1[0, 2] = (d - x) * z / m ** 3 + (d - x) * z / h ** 3
    H1[1, 0] = -(d - x) * (s - y) / m ** 3 + (d - x) * y / h ** 3
    H1[1, 1] = 1 / m + 1 / h - ((s - y) ** 2) / m ** 3 - (y ** 2) / h ** 3
    H1[1, 2] = (s - y) * z / m ** 3 - y * z / h ** 3
    H1[2, 0] = (d - x) * z / m ** 3 + (d - x) * z / h ** 3
    H1[2, 1] = (s - y) * z / m ** 3 - y * z / h ** 3
    H1[2, 2] = 1 / m + 1 / h - (z ** 2) / m ** 3 - (z ** 2) / h ** 3

    H2 = np.zeros((3, 3))
    H2[0, 0] = 1 / m + 1 / n - ((d - x) ** 2) / m ** 3 - (x ** 2) / n ** 3
    H2[0, 1] = -(d - x) * (s - y) / m ** 3 + x * (s - y) / n ** 3
    H2[0, 2] = (d - x) * z / m ** 3 - x * z / n ** 3
    H2[1, 0] = -(d - x) * (s - y) / m ** 3 + x * (s - y) / n ** 3
    H2[1, 1] = 1 / m + 1 / n - ((s - y) ** 2) / m ** 3 - ((s - y) ** 2) / n ** 3
    H2[1, 2] = (s - y) * z / m ** 3 + (s - y) * z / n ** 3
    H2[2, 0] = (d - x) * z / m ** 3 - x * z / n ** 3
    H2[2, 1] = (s - y) * z / m ** 3 + (s - y) * z / n ** 3
    H2[2, 2] = 1 / m + 1 / n - (z ** 2) / m ** 3 - (z ** 2) / n ** 3

    H3 = np.zeros((3, 3))
    H3[0, 0] = 1 / m + 1 / n + 1 / k + 1 / h - ((d - x) ** 2) / m ** 3 - (x ** 2) / n ** 3 - (x ** 2) / k ** 3 - (
                (d - x) ** 2) / h ** 3
    H3[0, 1] = -(d - x) * (s - y) / m ** 3 + x * (s - y) / n ** 3 - x * y / k ** 3 + (d - x) * y / h ** 3
    H3[0, 2] = (d - x) * z / m ** 3 - x * z / n ** 3 - x * z / k ** 3 + (d - x) * z / h ** 3
    H3[1, 0] = -(d - x) * (s - y) / m ** 3 + x * (s - y) / n ** 3 - x * y / k ** 3 + (d - x) * y / h ** 3
    H3[1, 1] = 1 / m + 1 / n + 1 / k + 1 / h - ((s - y) ** 2) / m ** 3 - ((s - y) ** 2) / n ** 3 - (y ** 2) / k ** 3 - (
                y ** 2) / h ** 3
    H3[1, 2] = (s - y) * z / m ** 3 + (s - y) * z / n ** 3 - y * z / k ** 3 - y * z / h ** 3
    H3[2, 0] = (d - x) * z / m ** 3 - x * z / n ** 3 - x * z / k ** 3 + (d - x) * z / h ** 3
    H3[2, 1] = (s - y) * z / m ** 3 + (s - y) * z / n ** 3 - y * z / k ** 3 - y * z / h ** 3
    H3[2, 2] = 1 / m + 1 / n + 1 / k + 1 / h - (z ** 2) / m ** 3 - (z ** 2) / n ** 3 - (z ** 2) / k ** 3 - (
                z ** 2) / h ** 3

    return H1 / R, H2 / R, H3 / R


def TrajRef(Xdep, Xfin, tfin, t1, t2, STADE, R, timestep=0.001):
    """
    Calcul de la trajectoire de référence
    entrée :    Xdep position de départ (vecteur colonne 3 composantes)
                Xfin position d'arrivée (vecteur colonne 3 composantes)
                tfin date d'arrivée (scalaire)
                t1 date à la fin de la première accélération (scalaire)
                t2 date au début du freinage (scalaire)
                STADE taille du stade (vecteur 3 composantes)
                R rayon des treuils (scalaire)
                dt temps d'échantillonnage (scalaire - optionnel)
    sortie : temps t (vecteur ligne taille N)
             vitXref position de référence (matrice 3 lignes N colonnes)
             posXref vitesse de référence (matrice 3 lignes N colonnes)
             vitThref vitesse angulaire de référence (matrice 3 lignes N colonnes)
             posThref position angulaire de référence (matrice 3 lignes N colonnes)
    """
    Nt = int((tfin - 0) / timestep + 1)
    t, dt = np.linspace(0, tfin, Nt, retstep=True)

    V0 = 2 * (Xfin - Xdep) / (tfin + (t2 - t1))
    vitXref = (t <= t1) * (V0 * t / t1) + \
              ((t > t1) * (t <= t2)) * V0 + \
              (t > t2) * (-V0 / (tfin - t2) * (t - t2) + V0)
    posXref = (t <= t1) * (V0 * t ** 2 / (2 * t1) + Xdep) + \
              ((t > t1) * (t <= t2)) * (V0 * (t - t1) + V0 * t1 / 2 + Xdep) + \
              (t > t2) * (-V0 / (2 * (tfin - t2)) * (t - t2) ** 2 + V0 * (t - t1) + V0 * t1 / 2 + Xdep)
    vitThref = np.zeros((3, Nt))
    for i in range(Nt):
        PXth = CalcPXTH(posXref[:, i], STADE, R)
        vitThref[:, i] = np.dot(PXth, vitXref[:, i])
        posThref = np.zeros((3, Nt))
    for k in range(3):
        posThref[k, :] = spint.cumtrapz(vitThref[k, :], initial=0) * dt

    return t, vitXref, posXref, vitThref, posThref


def CalcConsigne(t, time, vitXref, posXref, vitThref, posThref):
    """"
    Formation des signaux de consigne pour la régulation par interpolation 
    à partir du calcul de la trajectoire de référence
    entrée  : t date à laquelle on souhaite déterminer la consigne (scalaire)
              time temps (vecteur ligne taille N)
              vitXref position de référence (vecteur 3 lignes N colonnes)
              posXref vitesse de référence (vecteur 3 lignes N colonnes)
              vitThref vitesse angulaire de référence (vecteur 3 lignes N colonnes)
              posThref position angulaire de référence (vecteur 3 lignes N colonnes)
    sortie : X_pt0 vitesse de consigne (vecteur 3 colonnes)
             X0    position de consigne (vecteur 3 colonnes)
             Th_pt0 vitesse angulaire de consigne (vecteur 3 colonnes)
             Th0 position angulaire de consigne (vecteur 3 colonnes)
    """
    X_pt0 = np.zeros((3, 1))
    X0 = np.zeros((3, 1))
    Th0 = np.zeros((3, 1))
    Th_pt0 = np.zeros((3, 1))
    for k in range(3):
        X_pt0[k] = np.interp(t, time, vitXref[k, :])
        X0[k] = np.interp(t, time, posXref[k, :])
        Th_pt0[k] = np.interp(t, time, vitThref[k, :])
        Th0[k] = np.interp(t, time, posThref[k, :])
    return X_pt0, X0, Th_pt0, Th0


def EqDiff(Y, t, STADE, R, mA, Sinv, L, G, g, Kpos, Kvit, time, vitXref, posXref, vitThref, posThref, Fpert):
    """
    Equation différentielle régissant le système  (Y)'=f(Y,t)
    entrée : Y vecteur 12 composantes 
             t temps (scalaire)
             ... tous les arguments nécessaires au calcul
    sortie : Y' vecteur 12 composantes 
    
    """
    Ypt = np.zeros(12)
    X = Y[0:3, None]
    Xpt = Y[3:6, None]
    Th = Y[6:9, None]
    Thpt = Y[9:12, None]

    P = CalcPXTH(X, STADE, R)
    H1, H2, H3 = CalcHXTH(X, STADE, R)
    Xpt0, X0, Thpt0, Th0 = CalcConsigne(t, time, vitXref, posXref, vitThref, posThref)

    M = np.dot(P.transpose(), Sinv) / mA
    N = np.linalg.inv(np.eye(3) + np.dot(np.dot(M, G), P))
    A = np.array([[np.dot(np.dot(Xpt.transpose(), H1), Xpt)[0][0]], [np.dot(np.dot(Xpt.transpose(), H2), Xpt)[0][0]],
                  [np.dot(np.dot(Xpt.transpose(), H3), Xpt)[0][0]]])
    B = np.dot(L, Thpt)
    C = Kpos * (Th0 - Th)
    D = Kvit * (Thpt0 - Thpt)
    E = -np.array([[0], [0], [g]]) + Fpert(t) / mA

    Ypt[0:3] = Xpt.transpose()
    Ypt[6:9] = Thpt.transpose()
    Xsec = np.dot(N, -np.dot(M, A) - np.dot(M, B) + np.dot(M, C) + np.dot(M, D) + E)
    Thsec = np.dot(P, Xsec) + A
    Ypt[3:6] = Xsec.transpose()
    Ypt[9:12] = Thsec.transpose()

    return Ypt
