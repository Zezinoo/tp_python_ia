"""
=================================
      Tests LiDAR 2D — Avant
=================================

Quatre visualisations (avant uniquement, 90° autour de l'axe +x) :
1) cartésien — capteur idéal (sans bruit)
2) cartésien — capteur réel (avec bruit)
3) polaire   — capteur idéal (sans bruit)
4) polaire   — capteur réel (avec bruit)
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from math import isclose

from src.geometrie import Point
from src.optique import CercleOptique, RectangleOptique, SceneOptique
from src.lidar import Lidar2D

# ---------------------------------------------------------------------
# Réglages d'I/O visuelle
# ---------------------------------------------------------------------
SHOW = True
SAVE = False
OUT_DIR = "tests/sortie_poo3_forward"
if SAVE and not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

def banniere(t):
    print("\n" + "="*len(t)); print(t); print("="*len(t))

def afficher(fig, name):
    if SAVE:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=130, bbox_inches="tight")
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
# Scène d'exemple : « route » devant + obstacle circulaire devant
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
# LiDAR : 90° frontaux autour de l'axe +x
#   - angles relatifs = [0 .. champ_de_vision]
#   - on centre le cône autour de 0 en posant orientation = -champ_de_vision/2
# ---------------------------------------------------------------------
def lidar_forward(sigma=0.0):
    FOV = np.deg2rad(90.0)
    return Lidar2D(
        position=Point(0.0, 0.0),
        orientation=-FOV/2,     # balayage de -45° à +45° (autour de +x)
        champ_de_vision=FOV,
        resolution=181,         # ~0.5° par pas
        distance_max=20.0,
        sigma_bruit=sigma       # 0.0 (idéal) ou >0 (réel)
    )

# ---------------------------------------------------------------------
# 1) Cartésien — idéal (sans bruit)
# ---------------------------------------------------------------------
def test_01_cartesien_sans_bruit():
    banniere("POO3/Avant/01) LiDAR cartésien - sans bruit (idéal)")
    sc = scene_avant()
    lidar = lidar_forward(sigma=0.0)
    mesures = lidar.balayer(sc)
    pts = lidar.mesures_en_points(mesures)

    fig, ax = plt.subplots()
    # scène
    for f in sc.obtenir_figures():
        f.tracer(ax)
    # retours
    ax.scatter([p.obtenir_x() for p in pts],
               [p.obtenir_y() for p in pts],
               s=10, c="tab:red", alpha=0.9, label="retours LiDAR (idéal)")
    # capteur
    ax.scatter([lidar.position.obtenir_x()],
               [lidar.position.obtenir_y()],
               marker="*", s=140, c="black", label="LiDAR")
    # cône de balayage (indicatif)
    champ_de_vision = lidar.champ_de_vision
    th0, th1 = lidar.orientation, lidar.orientation + champ_de_vision
    r = 4.0
    ax.plot([0, r*np.cos(th0)], [0, r*np.sin(th0)], "k--", lw=0.8, alpha=0.6)
    ax.plot([0, r*np.cos(th1)], [0, r*np.sin(th1)], "k--", lw=0.8, alpha=0.6)

    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs + [(-0.5,-0.5,0.5,0.5)], ax)
    ax.legend()
    ax.set_title("LiDAR 2D — retour cartésien (avant 90°, sans bruit)")
    afficher(fig, "01_forward_cartesian_ideal")

# ---------------------------------------------------------------------
# 2) Cartésien — réel (avec bruit)
# ---------------------------------------------------------------------
def test_02_cartesien_avec_bruit():
    banniere("POO3/Avant/02) LiDAR cartésien — avec bruit (réel)")
    sc = scene_avant()
    lidar = lidar_forward(sigma=0.03)  # bruit ~3 cm
    mesures = lidar.balayer(sc)
    pts = lidar.mesures_en_points(mesures)

    fig, ax = plt.subplots()
    for f in sc.obtenir_figures():
        f.tracer(ax)
    ax.scatter([p.obtenir_x() for p in pts],
               [p.obtenir_y() for p in pts],
               s=10, c="tab:orange", alpha=0.9, label="retours LiDAR (bruité écart-type=3cm)")
    ax.scatter([lidar.position.obtenir_x()],
               [lidar.position.obtenir_y()],
               marker="*", s=140, c="black", label="LiDAR")

    champ_de_vision = lidar.champ_de_vision
    th0, th1 = lidar.orientation, lidar.orientation + champ_de_vision
    r = 4.0
    ax.plot([0, r*np.cos(th0)], [0, r*np.sin(th0)], "k--", lw=0.8, alpha=0.6)
    ax.plot([0, r*np.cos(th1)], [0, r*np.sin(th1)], "k--", lw=0.8, alpha=0.6)

    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs + [(-0.5,-0.5,0.5,0.5)], ax)
    ax.legend()
    ax.set_title("LiDAR 2D — retour cartésien (avant 90°, avec bruit)")
    afficher(fig, "02_forward_cartesian_noisy")

# ---------------------------------------------------------------------
# 3) Polaire — idéal (sans bruit)
# ---------------------------------------------------------------------
def test_03_polaire_sans_bruit():
    banniere("POO3/Avant/03) LiDAR polaire - sans bruit (idéal)")
    sc = scene_avant()
    lidar = lidar_forward(sigma=0.0)
    mesures = lidar.balayer(sc)

    # Filtrer les None
    angles = np.array([m.angle for m in mesures if m.distance is not None])
    distances = np.array([m.distance for m in mesures if m.distance is not None])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(angles, distances, ".", markersize=3, alpha=0.95)
    ax.set_theta_zero_location("E")  # 0 rad vers la droite
    ax.set_theta_direction(-1)       # angles croissants sens horaire (optionnel)
    ax.set_title("LiDAR 2D - vue polaire (avant 90°, sans bruit)")
    afficher(fig, "03_forward_polar_ideal")

# ---------------------------------------------------------------------
# 4) Polaire — réel (avec bruit)
# ---------------------------------------------------------------------
def test_04_polaire_avec_bruit():
    banniere("POO3/Avant/04) LiDAR polaire - avec bruit (réel)")
    sc = scene_avant()
    lidar = lidar_forward(sigma=0.03)
    mesures = lidar.balayer(sc)

    angles = np.array([m.angle for m in mesures if m.distance is not None])
    distances = np.array([m.distance for m in mesures if m.distance is not None])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(angles, distances, ".", markersize=3, alpha=0.95)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("LiDAR 2D - vue polaire (avant 90°, avec bruit écart-type=3cm)")
    afficher(fig, "04_forward_polar_noisy")
    
# ---------------------------------------------------------------------
# Lance tous les tests si appelé directement
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\nDébut des tests (LiDAR avant, 90°).")
    test_01_cartesien_sans_bruit()
    test_02_cartesien_avec_bruit()
    test_03_polaire_sans_bruit()
    test_04_polaire_avec_bruit()
    print("\nFin des tests.")
