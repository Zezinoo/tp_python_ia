# TP  Descente de gradient appliquée à la régression linéaire

Objectif : estimer les paramètres d'un modèle linéaire $\hat{y} = \theta_0 + \theta_1 x$ en
minimisant le coût (ici, l'erreur quadratique moyenne) avec l'algorithme de descente de gradient
par lot.

## Arborescence

```
tp8/
├── src/
│   ├── gradient_descent.py        # à compléter (coût, gradient, descente, tracés)
│   ├── animation_gd.py            # contours + trajectoire
│   └── __init__.py
├── tests/
│   ├── test_gradient_descent.py   # test de décroissance du coût
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Installation (recommandé)

```bash
python3 -m venv env_tp8
source env_tp8/bin/activate
pip install -r requirements.txt
```

## Ce que vous devez faire

1. Données : complétez `generate_data()` et associez-la avec `plot_data()` (visualisation).
2. Coût : codez `cost_function(theta0, theta1, X, y)`.
3. Gradient : codez `gradient(theta0, theta1, X, y)` (formules analytiques).
4. Descente de gradient : codez `gradient_descent(...)` (boucle, historique).
5. Tracés : ajoutez un graphe de `J(itération)` et la droite ajustée.
6. Contours + trajectoire : exécutez `animation_gd.py` pour tracer
   des lignes de niveau de $J(\theta_0, \theta_1) et la trajectoire suivie
   par la descente de gradient.
7. Influence de $\alpha$ : comparez plusieurs pas d'apprentissage.

## Lancer les tests

```bash
pytest -q
```

## Remarques

- Vectorisez avec *NumPy* (pas de boucles Python inutiles).
- Soignez les étiquettes, légendes, et grilles des figures pour la lisibilité.
- Commencez avec `alpha=0.05`, `n_iter=500`. Observez l'effet de $\alpha$.

