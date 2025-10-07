"""
=================================
         Tests LiDAR 2D
=================================

Deux séries de quatre visualisations avec un angle de vision avant de
90° autour de l'axe +x) :
Série A : scène simple
  1) cartésien - capteur idéal (sans bruit)
  2) cartésien - capteur réel (avec bruit)
  3) polaire   - capteur idéal (sans bruit)
  4) polaire   - capteur réel (avec bruit)

Série B : scène complexe (labyrinthe + obstacles)
  5) cartésien - capteur idéal (sans bruit)
  6) cartésien - capteur réel (avec bruit)
  7) polaire   - capteur idéal (sans bruit)
  8) polaire   - capteur réel (avec bruit)
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.geometrie import Point
from src.optique import CercleOptique, RectangleOptique, SceneOptique
from src.lidar import Lidar2D

# ---------------------------------------------------------------------
# Réglages d'E/S visuelle
# ---------------------------------------------------------------------
SHOW = True
SAVE = False
OUT_DIR = "tests/sortie_poo3"
if SAVE and not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

def banniere(t):
    print("\n" + "="*len(t)); print(t); print("="*len(t))

def afficher(fig, name):
    if SAVE:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.svg"), dpi=130, bbox_inches="tight")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)

def cadrer_axes_sur(bbs, ax, pad_ratio=0.12):
    xs = [b[0] for b in bbs] + [b[2] for b in bbs]
    ys = [b[1] for b in bbs] + [b[3] for b in bbs]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = pad_ratio * max(1.0, (x1 - x0) + (y1 - y0))
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)

# ---------------------------------------------------------------------
# A) Scène simple : "route" devant + obstacle circulaire devant
# ---------------------------------------------------------------------
def scene_avant():
    sc = SceneOptique()
    # "sol" et "plafond" (faibles épaisseurs) pour donner un contexte visuel
    sc.ajouter(RectangleOptique(Point(-5.0, -2.5), 16.0, 0.1))  # bas
    sc.ajouter(RectangleOptique(Point(-5.0,  2.4), 16.0, 0.1))  # haut
    # obstacle circulaire bien EN AVANT du LiDAR
    sc.ajouter(CercleOptique(Point(3.0, 0.0), 1.0))
    # mince paroi verticale encore plus loin
    sc.ajouter(RectangleOptique(Point(8.0, -2.0), 0.1, 4.0))
    return sc

# ---------------------------------------------------------------------
# B) Scène complexe : boîte + cloisons minces aléatoires + cercles
#    (reproductible en conservant une graine à 7)
# ---------------------------------------------------------------------
def scene_labyrinthe(seed: int = 7):
    rng = np.random.default_rng(seed)
    sc = SceneOptique()

    # Boîte extérieure
    x0, y0, x1, y1 = -8.0, -5.0, 8.0, 5.0
    eps = 0.12
    sc.ajouter(RectangleOptique(Point(x0, y0), x1-x0, eps))      # bas
    sc.ajouter(RectangleOptique(Point(x0, y1-eps), x1-x0, eps))  # haut
    sc.ajouter(RectangleOptique(Point(x0, y0), eps, y1-y0))      # gauche
    sc.ajouter(RectangleOptique(Point(x1-eps, y0), eps, y1-y0))  # droite

    # Cloisons minces (verticales/horizontales)
    for _ in range(6):
        if rng.random() < 0.5:
            x = rng.uniform(x0+1.0, x1-1.0)
            h = rng.uniform(1.0, 4.0)
            y = rng.uniform(y0+0.5, y1-h-0.5)
            sc.ajouter(RectangleOptique(Point(x, y), 0.10, h))
        else:
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

# ---------------------------------------------------------------------
# LiDAR : 90° frontaux autour de l'axe +x
#   - angles relatifs = [0 .. champ_de_vision]
#   - on centre le cône autour de 0 en posant orientation = -champ_de_vision/2
# ---------------------------------------------------------------------
def lidar_forward(position: Point, sigma=0.0):
    FOV = np.deg2rad(90.0)
    return Lidar2D(
        position=position,
        orientation=-FOV/2,     # balayage de -45° à +45° (autour de +x)
        champ_de_vision=FOV,
        resolution=181,         # ~0.5° par pas
        distance_max=40.0,
        sigma_bruit=sigma       # 0.0 (idéal) ou >0 (réel)
    )

def dessiner_cone(ax, origin: Point, theta0: float, theta1: float, r: float = 4.0):
    ox, oy = origin.obtenir_x(), origin.obtenir_y()
    ax.plot([ox, ox + r*np.cos(theta0)], [oy, oy + r*np.sin(theta0)], "k--", lw=0.8, alpha=0.6)
    ax.plot([ox, ox + r*np.cos(theta1)], [oy, oy + r*np.sin(theta1)], "k--", lw=0.8, alpha=0.6)

# =========================
# SÉRIE A - SCÈNE SIMPLE
# =========================
def test_01_cartesien_sans_bruit():
    banniere("POO3/A1) Scène simple (cartésien, sans bruit)")
    sc = scene_avant()
    lidar = lidar_forward(position=Point(0.0, 0.0), sigma=0.0)
    mesures = lidar.balayer(sc)
    pts = lidar.mesures_en_points(mesures)

    fig, ax = plt.subplots()
    for f in sc.obtenir_figures():
        f.tracer(ax)
    ax.scatter([p.obtenir_x() for p in pts],
               [p.obtenir_y() for p in pts],
               s=10, c="tab:red", alpha=0.9, label="retours LiDAR (idéal)")
    ax.scatter([lidar.position.obtenir_x()],
               [lidar.position.obtenir_y()],
               marker="*", s=140, c="black", label="LiDAR")

    th0 = lidar.orientation
    th1 = lidar.orientation + lidar.champ_de_vision
    dessiner_cone(ax, lidar.position, th0, th1)

    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs + [(-0.5,-0.5,0.5,0.5)], ax)
    ax.legend()
    ax.set_title("LiDAR 2D - scène simple (cartésien, 90°, sans bruit)")
    afficher(fig, "A01_forward_cartesian_ideal")

def test_02_cartesien_avec_bruit():
    banniere("POO3/A2) Scène simple (cartésien et avec bruit)")
    sc = scene_avant()
    lidar = lidar_forward(position=Point(0.0, 0.0), sigma=0.03)  # 3 cm
    mesures = lidar.balayer(sc)
    pts = lidar.mesures_en_points(mesures)

    fig, ax = plt.subplots()
    for f in sc.obtenir_figures():
        f.tracer(ax)
    ax.scatter([p.obtenir_x() for p in pts],
               [p.obtenir_y() for p in pts],
               s=10, c="tab:orange", alpha=0.9, label="retours LiDAR (sigma=3cm)")
    ax.scatter([lidar.position.obtenir_x()],
               [lidar.position.obtenir_y()],
               marker="*", s=140, c="black", label="LiDAR")

    th0 = lidar.orientation
    th1 = lidar.orientation + lidar.champ_de_vision
    dessiner_cone(ax, lidar.position, th0, th1)

    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs + [(-0.5,-0.5,0.5,0.5)], ax)
    ax.legend()
    ax.set_title("LiDAR 2D - scène simple (cartésien, 90°, avec bruit)")
    afficher(fig, "A02_forward_cartesian_noisy")

def test_03_polaire_sans_bruit():
    banniere("POO3/A3) Scène simple (polaire, sans bruit)")
    sc = scene_avant()
    lidar = lidar_forward(position=Point(0.0, 0.0), sigma=0.0)
    mesures = lidar.balayer(sc)

    angles = np.array([m.angle for m in mesures if m.distance is not None])
    distances = np.array([m.distance for m in mesures if m.distance is not None])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(angles, distances, ".", markersize=3, alpha=0.95)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("LiDAR 2D - scène simple (polaire, 90°, sans bruit)")
    afficher(fig, "A03_forward_polar_ideal")

def test_04_polaire_avec_bruit():
    banniere("POO3/A4) Scène simple (polaire, avec bruit)")
    sc = scene_avant()
    lidar = lidar_forward(position=Point(0.0, 0.0), sigma=0.03)
    mesures = lidar.balayer(sc)

    angles = np.array([m.angle for m in mesures if m.distance is not None])
    distances = np.array([m.distance for m in mesures if m.distance is not None])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(angles, distances, ".", markersize=3, alpha=0.95)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("LiDAR 2D - scène simple (polaire, 90°, sigma=3cm)")
    afficher(fig, "A04_forward_polar_noisy")

# =========================
# SÉRIE B - SCÈNE COMPLEXE
# =========================
def test_05_cartesien_sans_bruit_complexe():
    banniere("POO3/B5) Scène complexe (cartésien, sans bruit)")
    sc = scene_labyrinthe(seed=7)
    lidar = lidar_forward(position=Point(-6.5, 0.0), sigma=0.0)
    mesures = lidar.balayer(sc)
    pts = lidar.mesures_en_points(mesures)

    fig, ax = plt.subplots()
    for f in sc.obtenir_figures():
        f.tracer(ax)
    ax.scatter([p.obtenir_x() for p in pts],
               [p.obtenir_y() for p in pts],
               s=9, c="tab:red", alpha=0.9, label="retours LiDAR (idéal)")
    ax.scatter([lidar.position.obtenir_x()],
               [lidar.position.obtenir_y()],
               marker="*", s=140, c="black", label="LiDAR")

    th0 = lidar.orientation
    th1 = lidar.orientation + lidar.champ_de_vision
    dessiner_cone(ax, lidar.position, th0, th1)

    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs, ax)
    ax.legend()
    ax.set_title("LiDAR 2D - scène complexe (cartésien, 90°, sans bruit)")
    afficher(fig, "B05_forward_cartesian_ideal_complex")

def test_06_cartesien_avec_bruit_complexe():
    banniere("POO3/B6) Scène complexe (cartésien, avec bruit)")
    sc = scene_labyrinthe(seed=7)
    lidar = lidar_forward(position=Point(-6.5, 0.0), sigma=0.05)  # 5 cm
    mesures = lidar.balayer(sc)
    pts = lidar.mesures_en_points(mesures)

    fig, ax = plt.subplots()
    for f in sc.obtenir_figures():
        f.tracer(ax)
    ax.scatter([p.obtenir_x() for p in pts],
               [p.obtenir_y() for p in pts],
               s=9, c="tab:orange", alpha=0.9, label="retours LiDAR (sigma=5cm)")
    ax.scatter([lidar.position.obtenir_x()],
               [lidar.position.obtenir_y()],
               marker="*", s=140, c="black", label="LiDAR")

    th0 = lidar.orientation
    th1 = lidar.orientation + lidar.champ_de_vision
    dessiner_cone(ax, lidar.position, th0, th1)

    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs, ax)
    ax.legend()
    ax.set_title("LiDAR 2D - scène complexe (cartésien, 90°, avec bruit)")
    afficher(fig, "B06_forward_cartesian_noisy_complex")

def test_07_polaire_sans_bruit_complexe():
    banniere("POO3/B7) Scène complexe (polaire, sans bruit)")
    sc = scene_labyrinthe(seed=7)
    lidar = lidar_forward(position=Point(-6.5, 0.0), sigma=0.0)
    mesures = lidar.balayer(sc)

    angles = np.array([m.angle for m in mesures if m.distance is not None])
    distances = np.array([m.distance for m in mesures if m.distance is not None])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(angles, distances, ".", markersize=2.8, alpha=0.95)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("LiDAR 2D - scène complexe (polaire, 90°, sans bruit)")
    afficher(fig, "B07_forward_polar_ideal_complex")

def test_08_polaire_avec_bruit_complexe():
    banniere("POO3/B8) Scène complexe (polaire, avec bruit)")
    sc = scene_labyrinthe(seed=7)
    lidar = lidar_forward(position=Point(-6.5, 0.0), sigma=0.05)
    mesures = lidar.balayer(sc)

    angles = np.array([m.angle for m in mesures if m.distance is not None])
    distances = np.array([m.distance for m in mesures if m.distance is not None])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(angles, distances, ".", markersize=2.8, alpha=0.95)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("LiDAR 2D - scène complexe (polaire, 90°, sigma=5cm)")
    afficher(fig, "B08_forward_polar_noisy_complex")

# ---------------------------------------------------------------------
# Lance tous les tests si appelé directement
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\nDébut des tests (LiDAR 90°).")
    # Série A (simple)
    test_01_cartesien_sans_bruit()
    test_02_cartesien_avec_bruit()
    test_03_polaire_sans_bruit()
    test_04_polaire_avec_bruit()
    # Série B (complexe)
    test_05_cartesien_sans_bruit_complexe()
    test_06_cartesien_avec_bruit_complexe()
    test_07_polaire_sans_bruit_complexe()
    test_08_polaire_avec_bruit_complexe()
    print("\nFin des tests.")
