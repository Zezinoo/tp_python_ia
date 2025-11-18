#!/usr/bin/env python3

import numpy as np
import pytest

from src.gradient_descent import (
    generate_data,
    cost_function,
    gradient,
    gradient_descent,
)

def test_cost_decreases_first_steps():
    # Données
    X, y = generate_data(n=40, seed=123)

    alpha = 0.05
    n_iter = 100

    # On attend une décroissance globale du coût sur les premières itérations
    theta0, theta1, J_hist, theta_hist = gradient_descent(X, y, alpha=alpha, n_iter=n_iter)

    assert len(J_hist) >= 5, "Historique trop court"
    assert J_hist[0] > J_hist[4], "Le coût ne décroît pas sur les premières itérations"

def test_gradients_match_numeric_roughly():
    # Test de cohérence grossier: gradient analytique vs estimation numérique
    X, y = generate_data(n=30, seed=7)
    theta0, theta1 = 0.8, 1.7

    # analytique
    g0, g1 = gradient(theta0, theta1, X, y)

    # numérique (différences finies)
    eps = 1e-5
    J = lambda t0, t1: cost_function(t0, t1, X, y)
    g0_num = (J(theta0 + eps, theta1) - J(theta0 - eps, theta1)) / (2 * eps)
    g1_num = (J(theta0, theta1 + eps) - J(theta0, theta1 - eps)) / (2 * eps)

    assert np.isfinite(g0) and np.isfinite(g1)
    assert np.allclose([g0, g1], [g0_num, g1_num], rtol=5e-2, atol=5e-2)
