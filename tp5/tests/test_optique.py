# tests/test_optique.py
# ============================================================
# Tests TP POO n°2 — Optique
# - Affiche une figure par test (SHOW=True)
# - Peut enregistrer les figures (SAVE=True)
# ============================================================
import os
from math import isclose, pi
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.optique import (
    Point, Vecteur2D, Transformation2D,
    CercleOptique, RectangleOptique,
    SceneOptique, Rayon, reflechir_direction
)

# ---------------------------------------------------------------------
# Réglages d'I/O visuelle
# ---------------------------------------------------------------------
SHOW = True
SAVE = False
OUT_DIR = "tests/sortie_poo2"
if SAVE and not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

def approx(a, b, rel=1e-7, abs_=1e-9): 
    return isclose(a, b, rel_tol=rel, abs_tol=abs_)

def banniere(t): 
    print("\n" + "="*len(t)); print(t); print("="*len(t))

def afficher(fig, name):
    if SAVE:
        # Remplacez par .png si vous préférez
        fig.savefig(os.path.join(OUT_DIR, f"{name}.svg"),
                    dpi=120, bbox_inches="tight")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)

def cadrer_axes_sur(bbs, ax, pad_ratio=0.15):
    """Ajuste les bornes du graphe pour contenir les boîtes englobantes fournies."""
    xs = [b[0] for b in bbs] + [b[2] for b in bbs]
    ys = [b[1] for b in bbs] + [b[3] for b in bbs]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = pad_ratio * max(1.0, (x1 - x0) + (y1 - y0))
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)

def dessiner_vecteur(ax, origine, v, **kw):
    """Trace une flèche représentant le vecteur v depuis 'origine' (Point)."""
    ax.quiver(origine.obtenir_x(), origine.obtenir_y(), v.x, v.y,
              angles='xy', scale_units='xy', scale=1, **kw)

def dessiner_segment(ax, p0: Point, p1: Point, **kw):
    ax.plot([p0.obtenir_x(), p1.obtenir_x()],
            [p0.obtenir_y(), p1.obtenir_y()], **kw)

# ---------------------------------------------------------------------
# 01) Réflexion locale
# ---------------------------------------------------------------------
def test_01_reflexion_locale():
    banniere("POO2/01) Réflexion locale — formule d'optique (visualisée)")
    d = Vecteur2D(1, 0).normaliser()     # direction incidente (-->)
    n = Vecteur2D(-1, 0)                 # normale locale (<--), déjà unitaire
    dr = reflechir_direction(d, n)       # direction réfléchie (<--)

    # Asserts numériques simples
    assert approx(dr.x, -1.0) and approx(dr.y, 0.0)

    # Visuel
    fig, ax = plt.subplots()
    O = Point(0, 0)
    dessiner_vecteur(ax, O, d, color="tab:blue", width=0.008, label="incident d")
    dessiner_vecteur(ax, O, n, color="tab:green", width=0.008, label="normale n")
    dessiner_vecteur(ax, O, dr, color="tab:orange", width=0.008, label="réfléchi d'")
    ax.legend()
    ax.set_title("Réflexion locale : d (bleu), n (vert), d' (orange)")
    cadrer_axes_sur([(-1.2, -1.0, 1.2, 1.0)], ax)
    afficher(fig, "01_reflexion_locale")

# ---------------------------------------------------------------------
# 02) Cercle — intersection rayon
# ---------------------------------------------------------------------
def test_02_cercle_intersection():
    banniere("POO2/02) CercleOptique — intersection rayon (visualisée)")
    c = CercleOptique(Point(0, 0), 1)
    ray = Rayon(Point(-2, 0), Vecteur2D(1, 0))
    res = c.intersection_rayon(ray.origine, ray.direction_unitaire())
    assert res is not None
    t, P, n = res

    # Vérifications numériques
    assert approx(t, 1.0)
    assert approx(P.obtenir_x(), -1.0) and approx(P.obtenir_y(), 0.0)

    fig, ax = plt.subplots()
    c.tracer(ax)

    # Coordonnées du rayon
    xs = [ray.origine.obtenir_x(), P.obtenir_x()]
    ys = [ray.origine.obtenir_y(), P.obtenir_y()]

    # Segment du rayon (tirets fins)
    ax.plot(xs, ys, "--", lw=1.0, color="tab:blue", label="rayon")

    # Départ et arrivée
    ax.scatter(xs[0], ys[0], marker="*", s=120, color="tab:blue", edgecolors="black")
    ax.scatter(xs[-1], ys[-1], marker="X", s=80, color="tab:blue", edgecolors="black")

    # Normale
    dessiner_vecteur(ax, P, n, color="tab:orange", width=0.008, label="normale")

    ax.legend()
    ax.set_title("Cercle — intersection du rayon")
    cadrer_axes_sur([c.boite_englobante(), (-2.5, -1.5, 2.5, 1.5)], ax)
    afficher(fig, "02_cercle_intersection")

# ---------------------------------------------------------------------
# 03) Rectangle — intersection rayon
# ---------------------------------------------------------------------
def test_03_rectangle_intersection():
    banniere("POO2/03) RectangleOptique — intersection rayon (visualisée)")
    r = RectangleOptique(Point(0, 0), 2, 1)
    ray = Rayon(Point(-1, 0.5), Vecteur2D(1, 0))
    res = r.intersection_rayon(ray.origine, ray.direction_unitaire())
    assert res is not None
    t, P, n = res

    fig, ax = plt.subplots()
    r.tracer(ax)

    # Coordonnées
    xs = [ray.origine.obtenir_x(), P.obtenir_x()]
    ys = [ray.origine.obtenir_y(), P.obtenir_y()]

    # Rayon
    ax.plot(xs, ys, "--", lw=1.0, color="tab:blue", label="rayon")
    ax.scatter(xs[0], ys[0], marker="*", s=120, color="tab:blue", edgecolors="black")
    ax.scatter(xs[-1], ys[-1], marker="X", s=80, color="tab:blue", edgecolors="black")

    # Normale
    dessiner_vecteur(ax, P, n, color="tab:orange", width=0.008, label="normale")

    ax.legend()
    ax.set_title("Rectangle — intersection du rayon")
    cadrer_axes_sur([r.boite_englobante(), (-1.5, -0.5, 3.0, 1.5)], ax)
    afficher(fig, "03_rectangle_intersection")

# ---------------------------------------------------------------------
# 04) SceneOptique — premier impact
# ---------------------------------------------------------------------
def test_04_premier_impact():
    banniere("POO2/04) SceneOptique — premier_impact() (visualisé)")
    sc = SceneOptique()
    sc.ajouter(CercleOptique(Point(0, 0), 1))
    sc.ajouter(RectangleOptique(Point(2, -1), 2, 1))

    ray = Rayon(Point(-2, 0.2), Vecteur2D(1, 0.05))
    hit = sc.premier_impact(ray)
    assert hit is not None and hit.t > 0

    fig, ax = plt.subplots()
    for f in sc.obtenir_figures():
        f.tracer(ax)

    # Rayon jusqu'à l'impact
    xs = [ray.origine.obtenir_x(), hit.point.obtenir_x()]
    ys = [ray.origine.obtenir_y(), hit.point.obtenir_y()]
    ax.plot(xs, ys, "--", lw=1.0, color="tab:blue", label="rayon")
    ax.scatter(xs[0], ys[0], marker="*", s=120, color="tab:blue", edgecolors="black")
    ax.scatter(xs[-1], ys[-1], marker="X", s=80, color="tab:blue", edgecolors="black")

    # Normale
    dessiner_vecteur(ax, hit.point, hit.normale, color="tab:orange", width=0.008, label="normale")

    ax.legend()
    ax.set_title("SceneOptique — premier impact du rayon")
    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs, ax)
    afficher(fig, "04_scene_premier_impact")

# ---------------------------------------------------------------------
# 05) SceneOptique — suivre_rayon (rebonds)
# ---------------------------------------------------------------------
def test_05_suivre_rayon():
    banniere("POO2/05) SceneOptique — suivre_rayon() (rebonds visualisés)")
    sc = SceneOptique.scene_exemple()

    # Trois rayons
    rays = [
        Rayon(Point(-3.0, 0.0),  Vecteur2D(1.0,  0.30)),
        Rayon(Point(-2.0, -3.14159265/4), Vecteur2D(1.0,  0.50)),
        Rayon(Point(-3.0, 1.5),  Vecteur2D(1.0, -0.25)),
    ]
    polylignes = [sc.suivre_rayon(r, nb_rebonds_max=10) for r in rays]
    for poly in polylignes:
        assert isinstance(poly, list) and len(poly) >= 2

    # Visuel : scène + rayons
    fig, ax = plt.subplots()
    for f in sc.obtenir_figures():
        f.tracer(ax)

    couleurs = ["tab:blue", "tab:orange", "tab:green"]

    for i, poly in enumerate(polylignes):
        col = couleurs[i]
        if i == 1:
            pass
        # Coordonnées
        xs = [p.obtenir_x() for p in poly]
        ys = [p.obtenir_y() for p in poly]

        # 1) Segment du rayon (tirets fins)
        ax.plot(xs, ys, linestyle="--", lw=1.2, color=col, alpha=1.0,
                label=f"rayon {i+1}", zorder=2)

        # 2) Marqueurs d’impacts (petits ronds)
        ax.plot(xs[1:-1], ys[1:-1], "o", ms=4, color=col, alpha=0.9, zorder=3)

        # 3) Point de départ (étoile bien visible)
        ax.scatter(xs[0], ys[0], marker="*", s=120, color=col, edgecolors="black",
                   linewidths=0.8, zorder=4)
        ax.annotate("", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=col)

        # 4) Point d’arrivée (dernier point) — croix épaisse
        ax.scatter(xs[-1], ys[-1], marker="X", s=80, color=col, edgecolors="black",
                   linewidths=0.8, zorder=4)
        ax.annotate("", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -10), fontsize=9, color=col)

        # (Optionnel) Petite flèche à la toute fin pour rappeler la direction
        if len(xs) >= 2:
            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
            ax.quiver(xs[-2], ys[-2], dx, dy, angles="xy", scale_units="xy", scale=1,
                      width=0.004, headwidth=4, headlength=6, headaxislength=5,
                      color=col, alpha=0.9, zorder=3)

    # Cadre et légende
    bbs = [f.boite_englobante() for f in sc.obtenir_figures()]
    cadrer_axes_sur(bbs, ax)
    ax.legend()
    ax.set_title("SceneOptique — rebonds de plusieurs rayons (départ *, sortie X)")
    afficher(fig, "05_scene_rebonds")

# ---------------------------------------------------------------------
# Lance tous les tests si appelé directement
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\nDébut des tests")
    #test_01_reflexion_locale()
    #test_02_cercle_intersection()
    #test_03_rectangle_intersection()
    #test_04_premier_impact()
    test_05_suivre_rayon()
    print("\nFin des tests !")
