#!/usr/bin/env python3

"""
Contours de J(theta0, theta1) + trajectoire prise par la descente de gradient.
Compléter le(s) TODO.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import generate_data, cost_function, gradient_descent

def make_contours_and_path(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    n_iter: int = 200,
    t0_bounds=(0.0, 3.0),
    t1_bounds=(0.0, 4.0),
    grid_size=120,
    show=True,
):
    """
    1) Exécute la descente de gradient pour obtenir theta_hist
    2) Calcule J sur une grille (theta0, theta1)
    3) Trace les lignes de niveau + la trajectoire (theta0_k, theta1_k)
    """
    # ---------- 1) Trajectoire par descente de gradient ----------
    theta_0 , theta_1 , J_hist , theta_hist = gradient_descent(X , y , alpha = alpha , n_iter= n_iter) 

    # ---------- 2) Grille de coût --------------------------------
    t0 = np.linspace(t0_bounds[0], t0_bounds[1], grid_size)
    t1 = np.linspace(t1_bounds[0], t1_bounds[1], grid_size)
    T0, T1 = np.meshgrid(t0, t1)
    Z = np.zeros_like(T0)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            Z[i, j] = cost_function(T0[i, j], T1[i, j], X, y)

    # ---------- 3) Tracé -----------------------------------------
    plt.figure()
    CS = plt.contour(T0, T1, Z, levels=30)
    plt.clabel(CS, inline=1, fontsize=8)
    plt.plot(theta_hist[:, 0], theta_hist[:, 1], "ro-", ms=3, lw=1, label="trajectoire")
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.title("Descente de gradient - courbes de niveau de J et trajectoire")
    plt.grid(True); plt.legend()
    if show: plt.show()


if __name__ == "__main__":
    X, y = generate_data()
    make_contours_and_path(X = X , y = y)
    # Démo minimale (quand le TODO sera fait)
    pass
