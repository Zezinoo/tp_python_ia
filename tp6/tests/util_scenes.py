# tests/utils_scenes.py
import numpy as np
from src.geometrie import Point
from src.optique import CercleOptique, RectangleOptique, SceneOptique

def scene_labyrinthe(seed: int = 42) -> SceneOptique:
    rng = np.random.default_rng(seed)
    sc = SceneOptique()

    # Boîte extérieure
    x0, y0, x1, y1 = -8.0, -5.0, 8.0, 5.0
    eps = 0.12
    sc.ajouter(RectangleOptique(Point(x0, y0), x1-x0, eps))      # bas
    sc.ajouter(RectangleOptique(Point(x0, y1-eps), x1-x0, eps))  # haut
    sc.ajouter(RectangleOptique(Point(x0, y0), eps, y1-y0))      # gauche
    sc.ajouter(RectangleOptique(Point(x1-eps, y0), eps, y1-y0))  # droite

    # Cloisons verticales/horizontales (largeur très fine)
    for _ in range(6):
        if rng.random() < 0.5:  # cloison verticale
            x = rng.uniform(x0+1.0, x1-1.0)
            h = rng.uniform(1.0, 4.0)
            y = rng.uniform(y0+0.5, y1-h-0.5)
            sc.ajouter(RectangleOptique(Point(x, y), 0.10, h))
        else:                   # cloison horizontale
            y = rng.uniform(y0+1.0, y1-1.0)
            w = rng.uniform(1.0, 6.0)
            x = rng.uniform(x0+0.5, x1-w-0.5)
            sc.ajouter(RectangleOptique(Point(x, y), w, 0.10))

    # Obstacles circulaires
    for _ in range(7):
        r = rng.uniform(0.25, 0.9)
        cx = rng.uniform(x0+1.0, x1-1.0)
        cy = rng.uniform(y0+1.0, y1-1.0)
        sc.ajouter(CercleOptique(Point(cx, cy), r))

    return sc


def scene_aleatoire(bounds=(-10, -6, 10, 6), n_rects=12, n_cercles=12, seed=1337) -> SceneOptique:
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bounds
    sc = SceneOptique()

    # Cadre
    eps = 0.12
    sc.ajouter(RectangleOptique(Point(x0, y0), x1-x0, eps))
    sc.ajouter(RectangleOptique(Point(x0, y1-eps), x1-x0, eps))
    sc.ajouter(RectangleOptique(Point(x0, y0), eps, y1-y0))
    sc.ajouter(RectangleOptique(Point(x1-eps, y0), eps, y1-y0))

    # Rectangles minces (murs / obstacles)
    for _ in range(n_rects):
        w = rng.uniform(0.3, 3.0)
        h = rng.uniform(0.08, 0.3) if rng.random() < 0.5 else rng.uniform(0.08, 0.3)  # minces
        if rng.random() < 0.5:
            w, h = h, w
        x = rng.uniform(x0+1.0, x1-1.0-w)
        y = rng.uniform(y0+1.0, y1-1.0-h)
        sc.ajouter(RectangleOptique(Point(x, y), w, h))

    # Cercles
    for _ in range(n_cercles):
        r = rng.uniform(0.2, 1.0)
        cx = rng.uniform(x0+1.0, x1-1.0)
        cy = rng.uniform(y0+1.0, y1-1.0)
        sc.ajouter(CercleOptique(Point(cx, cy), r))

    return sc
