#!/usr/bin/env python3

"""
TP: Descente de gradient appliquée à la régression linéaire.
À COMPLÉTER : génération de données, fonction de coût,
              gradient, descente de gradient, et tracés.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) Génération des données (À COMPLÉTER)
# ---------------------------------------------------------
def generate_data(n: int = 50, seed: int = 42):
    """
    Génère des données (X, y) bruitées selon une droite cible ~ (theta0 ~ 1.5, theta1 ~ 2.5).
    Retour:
        X: np.ndarray shape (n,)
        y: np.ndarray shape (n,)
    """
    # TODO: initialisez le générateur de nombres aléatoires et générez X régulièrement dans [0, 10]
    # TODO: générez y = 2.5 * X + 1.5 + bruit gaussien (écart-type ~ 1.0)
    raise NotImplementedError("À implémenter: generate_data")

def plot_data(X: np.ndarray, y: np.ndarray, show: bool = True, ax=None):
    """
    Affiche le nuage de points (X, y).
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(X, y, s=25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Nuage de points")
    ax.grid(True)
    if show:
        plt.show()

# ---------------------------------------------------------
# 2) Fonction de coût (À COMPLÉTER)
# ---------------------------------------------------------
def cost_function(theta0: float, theta1: float, X: np.ndarray, y: np.ndarray) -> float:
    """
    J(theta0, theta1) = (1/(2n)) * \Sigma_i (theta0 + theta1*xi - yi)^2
    - Vectorisez avec NumPy.
    """
    # TODO
    raise NotImplementedError("À implémenter : cost_function()")


# ---------------------------------------------------------
# 3) Gradient analytique (À COMPLÉTER)
# ---------------------------------------------------------
def gradient(theta0: float, theta1: float, X: np.ndarray, y: np.ndarray):
    """
    Retourne (dJ/dtheta0, dJ/dtheta1) avec :
      dJ/dtheta0 = (1/n) * \Sigma_i (theta0 + theta1*xi - yi)
      dJ/dtheta1 = (1/n) * \Sigma_i (theta0 + theta1*xi - yi) * xi
    - Vectorisez avec NumPy.
    """
    # TODO
    raise NotImplementedError("À implémenter : gradient()")


# ---------------------------------------------------------
# 4) Descente de gradient (À COMPLÉTER)
# ---------------------------------------------------------
def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    n_iter: int = 500,
    tol: float = 1e-8,
):
    """
    Boucle de descente de gradient par lot:
      - init theta0, theta1 = 0
      - à chaque itération: calculez le gradient, mettez à jour, calculez J et stockez
      - critère d'arrêt optionnel: |J_k - J_{k-1}| < tol
    Retourne:
      theta0, theta1, J_hist (np.ndarray), theta_hist (np.ndarray shape (k,2))
    """
    # TODO
    raise NotImplementedError("À implémenter: gradient_descent")


# ---------------------------------------------------------
# 5) Tracés d'analyse
# ---------------------------------------------------------
def plot_fit(X: np.ndarray, y: np.ndarray, theta0: float, theta1: float, show: bool = True):
    """
    Affiche les données et la droite ajustée y_hat = theta0 + theta1 x.
    """
    xx = np.linspace(np.min(X), np.max(X), 200)
    yy = theta0 + theta1 * xx
    fig, ax = plt.subplots()
    ax.scatter(X, y, s=25, label="Données")
    ax.plot(xx, yy, label=r"$\hat{y}=\theta_0+\theta_1 x$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ajustement linéaire (descente de gradient)")
    ax.grid(True)
    ax.legend()
    if show:
        plt.show()

def plot_cost_history(J_hist: np.ndarray, show: bool = True):
    """
    Affiche l'évolution du coût J par itération.
    """
    fig, ax = plt.subplots()
    ax.plot(J_hist)
    ax.set_xlabel("Itérations")
    ax.set_ylabel(r"Coût $J(\theta_0,\theta_1)$")
    ax.set_title("Descente de gradient - évolution du coût")
    ax.grid(True)
    if show:
        plt.show()

# ---------------------------------------------------------
# 6) Influence du pas d'apprentissage (À COMPLÉTER)
# ---------------------------------------------------------
def compare_alphas(X: np.ndarray, y: np.ndarray, alphas=(0.001, 0.01, 0.05, 0.2)):
    """
    Compare l'évolution de J pour différents alpha.
    """
    # TODO:
    # for a in alphas:
    #     _, _, J_hist, _ = gradient_descent(X, y, alpha=a, n_iter=500)
    #     plt.plot(J_hist, label=f"alpha={a}")
    # plt.xlabel("Itérations"); plt.ylabel("J"); plt.legend(); plt.grid(True); plt.show()
    raise NotImplementedError("À implémenter: compare_alphas")


# ---------------------------------------------------------
# 7) Exemple d'utilisation
# ---------------------------------------------------------
if __name__ == "__main__":
    # Petit scénario de test manuel (à commenter/décommenter selon avancement)
    pass
