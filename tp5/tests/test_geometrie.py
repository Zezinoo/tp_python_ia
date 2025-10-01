# tests/test_geometrie.py
# ============================================================
# Tests TP POO n°1 — Géométrie 2D
# - Affiche une figure par test (SHOW=True)
# - Peut enregistrer les figures (SAVE=True)
# ============================================================

import os
from math import isclose, pi
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.geometrie import (
    Vecteur2D, Point, Transformation2D,
    Cercle, Rectangle, Scene
)

# ---------------- Réglages I/O visuelle ---------------------
SHOW = True      # True: afficher les figures / False: ne pas afficher
SAVE = False     # True: enregistrer les figures
OUT_DIR = "tests/sortie_poo1"
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
    """
    `bbs` : liste de boites englobantes (xmin,ymin,xmax,ymax) ou tuples (x0,y0,x1,y1).
    Ajuste les limites du graphe pour tout contenir, avec une marge.
    """
    xs = []
    ys = []
    for (x0,y0,x1,y1) in bbs:
        xs += [x0, x1]; ys += [y0, y1]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = pad_ratio * max(1.0, (x1 - x0) + (y1 - y0))
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)

# ------------------------------------------------------------
# 01) Vecteur2D — opérations de base (avec tracé)
# ------------------------------------------------------------
def test_01_vecteur2d():
    banniere("POO1/01) Vecteur2D — opérations de base")
    v = Vecteur2D(3, 4)
    assert approx(v.norme(), 5.0)

    vn = v.normaliser()
    assert approx(vn.norme(), 1.0)
    assert approx(v.ajouter(Vecteur2D(1, -2)).x, 4)
    assert approx(v.soustraire(Vecteur2D(2, 1)).y, 3)
    assert approx(v.multiplier(2).x, 6)
    assert approx(v.produit_scalaire(Vecteur2D(-4, 1)), -8)

    # On affiche : vecteur brut + normalisé + cercle unité
    fig, ax = plt.subplots()
    circ = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", linewidth=1.2, alpha=0.6, color="gray")
    ax.add_patch(circ)

    # Vecteur original
    ax.quiver(0, 0, v.x, v.y, angles="xy", scale_units="xy", scale=1,
              color="tab:blue", width=0.010, label="v (brut)")

    # Vecteur normalisé
    ax.quiver(0, 0, vn.x, vn.y, angles="xy", scale_units="xy", scale=1,
              color="tab:orange", width=0.010, label="normaliser(v)")

    # Marque les extrémités
    ax.scatter([v.x], [v.y], s=50, color="tab:blue", edgecolors="black", linewidths=0.6)
    ax.scatter([vn.x], [vn.y], s=50, color="tab:orange", edgecolors="black", linewidths=0.6)

    # Cadre
    L = max(1.0, v.norme()) * 1.4
    ax.set_xlim(-0.2, L); ax.set_ylim(-0.2, L)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    ax.set_title("Vecteur2D : v (bleu) / normaliser(v) (orange)")

    afficher(fig, "01_vecteur2d")

# ------------------------------------------------------------
# 02) Point — translation (avec tracé)
# ------------------------------------------------------------
def test_02_point():
    banniere("POO1/02) Point - translation")
    p = Point(0.5, -0.2)
    x0, y0 = p.obtenir_x(), p.obtenir_y()
    p.translater(0.8, 0.6)
    x1, y1 = p.obtenir_x(), p.obtenir_y()
    assert approx(x1 - x0, 0.8) and approx(y1 - y0, 0.6)

    # On affiche : point avant (bleu) / après (orange)
    fig, ax = plt.subplots()
    ax.scatter([x0], [y0], c="tab:blue", s=60, label="avant")
    ax.scatter([x1], [y1], c="tab:orange", s=60, label="après")
    ax.plot([x0, x1], [y0, y1], "--", lw=1.0, color="gray")
    ax.legend(); ax.set_title("Point : translation (avant --> après)")
    cadrer_axes_sur([(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))], ax)
    afficher(fig, "02_point_translation")

# ------------------------------------------------------------
# 03) Transformation2D — rotation/translation/mise à l'échelle/compose
# ------------------------------------------------------------
def test_03_transformation2d():
    banniere("POO1/03) Transformation2D - rotation / translation / mise à l'échelle / composition")
    from math import pi

    # Rectangle de départ
    r = Rectangle(Point(0, 0), 2, 1)

    # Transformations
    Trot = Transformation2D.rotation(pi/2)          # +90°
    Ttr  = Transformation2D.translation(2, -1)      # translation
    Tsc  = Transformation2D.mise_echelle(1.5, 0.5)  # mise à l'échelle
    Tcmp = Ttr.composer(Trot)                       # composition Ttr ∘ Trot

    # Rectangles transformés
    r_rot = r.transformer(Trot)
    r_tr  = r.transformer(Ttr)
    r_scl = r.transformer(Tsc)
    r_cmp = r.transformer(Tcmp)

    # On affiche
    fig, ax = plt.subplots()

    # 1) Original — bleu ÉPAIS
    r.tracer(ax)
    ax.patches[-1].set_edgecolor("tab:blue")
    ax.patches[-1].set_linewidth(2.6)
    ax.patches[-1].set_linestyle("-")

    # 2) Transformés — couleurs/styles distincts
    r_rot.tracer(ax); ax.patches[-1].set_edgecolor("tab:orange"); ax.patches[-1].set_linestyle("--")
    r_tr.tracer(ax);  ax.patches[-1].set_edgecolor("tab:green");  ax.patches[-1].set_linestyle(":")
    r_scl.tracer(ax); ax.patches[-1].set_edgecolor("tab:purple"); ax.patches[-1].set_linestyle("-.")
    r_cmp.tracer(ax); ax.patches[-1].set_edgecolor("tab:red");    ax.patches[-1].set_linestyle("-")

    # Cadre et habillage
    bbs = [x.boite_englobante() for x in (r, r_rot, r_tr, r_scl, r_cmp)]
    cadrer_axes_sur(bbs, ax)
    ax.set_title("Transformation2D : original (bleu épais) et versions transformées")
    ax.grid(True, linestyle=":", alpha=0.5)
    afficher(fig, "03_transformation2d")

# ------------------------------------------------------------
# 04) Cercle — aire/boîte englobante/contient (avec tracé)
# ------------------------------------------------------------
def test_04_cercle():
    banniere("POO1/04) Cercle - aire/boîte englobante/contient (avec tracé)")
    c = Cercle(Point(0, 0), 1.2)
    assert c.perimetre() > 0 and c.aire() > 0
    xmin, ymin, xmax, ymax = c.boite_englobante()
    assert approx(xmax - xmin, 2*c.obtenir_rayon())
    assert approx(ymax - ymin, 2*c.obtenir_rayon())

    p_in  = Point(0.5, 0.1)
    p_out = Point(2.0, 0.0)
    assert c.contient(p_in)
    assert not c.contient(p_out)

    # On affiche
    fig, ax = plt.subplots()
    c.tracer(ax)

    # Points : plus visibles + annotation
    ax.scatter([p_in.obtenir_x()],  [p_in.obtenir_y()],
               c="tab:red", s=70, edgecolors="black", linewidths=0.6, label="dedans")
    ax.scatter([p_out.obtenir_x()], [p_out.obtenir_y()],
               c="tab:orange", s=70, edgecolors="black", linewidths=0.6, label="dehors")
    ax.annotate("dedans", (p_in.obtenir_x(), p_in.obtenir_y()),
                textcoords="offset points", xytext=(6, 6), fontsize=9, color="tab:red")
    ax.annotate("dehors", (p_out.obtenir_x(), p_out.obtenir_y()),
                textcoords="offset points", xytext=(6, -10), fontsize=9, color="tab:orange")

    # Boîte englobante du cercle (trait fin noir)
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            "-", lw=1.0, color="black", alpha=0.6, label="boite englobante cercle")

    ax.set_title("Cercle : inclusion & boîte englobante")
    ax.grid(True, linestyle=":", alpha=0.5)

    # POO1/Cadrage : inclure le cercle ET les points de test
    cadrer_axes_sur([
        c.boite_englobante(),
        (p_in.obtenir_x(),  p_in.obtenir_y(),  p_in.obtenir_x(),  p_in.obtenir_y()),
        (p_out.obtenir_x(), p_out.obtenir_y(), p_out.obtenir_x(), p_out.obtenir_y()),
    ], ax)

    ax.legend()
    afficher(fig, "04_cercle")

# ------------------------------------------------------------
# 05) Rectangle — aire/boite englobante/contient (avec tracé)
# ------------------------------------------------------------
def test_05_rectangle():
    banniere("POO1/05) Rectangle — aire/boîte englobante/contient (avec tracé)")
    r = Rectangle(Point(-1, -0.5), 2, 1)
    assert r.perimetre() > 0 and r.aire() > 0
    xmin, ymin, xmax, ymax = r.boite_englobante()
    assert approx(xmax - xmin, r.obtenir_largeur())
    assert approx(ymax - ymin, r.obtenir_hauteur())

    p_in  = Point(0.0, 0.0)
    p_out = Point(1.1, 0.6)
    assert r.contient(p_in)
    assert not r.contient(p_out)

    # On affiche
    fig, ax = plt.subplots()
    r.tracer(ax)
    ax.scatter([p_in.obtenir_x()],  [p_in.obtenir_y()],  c="tab:red",    s=40, label="dedans")
    ax.scatter([p_out.obtenir_x()], [p_out.obtenir_y()], c="tab:orange", s=40, label="dehors")
    # bbox (en trait fin noir)
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin], "-", lw=1.0, color="black", alpha=0.6, label="boîte englobante")
    ax.legend(); ax.set_title("Rectangle : inclusion & boîte englobante")
    cadrer_axes_sur([r.boite_englobante()], ax)
    afficher(fig, "05_rectangle")

# ------------------------------------------------------------
# 06) Scene — aire/perimètre/bbox/translation (avec tracé)
# ------------------------------------------------------------
def test_06_scene():
    banniere("POO1/06) Scene — aire/perimètre/boîte englobante/translation (avec tracé)")
    s = Scene()
    c = Cercle(Point(0, 0), 1.0)
    r = Rectangle(Point(1.0, 0.5), 2.0, 1.0)
    s.ajouter(c); s.ajouter(r)

    # Aire et périmètre totaux (valeurs > 0, pas de formules imposées ici)
    A0, P0 = s.aire_totale(), s.perimetre_total()
    assert A0 > 0 and P0 > 0
    bb0 = s.boite_scene()

    # Translation globale
    dx, dy = 0.8, -0.4
    s2 = Scene()
    for f in s.obtenir_figures():
        s2.ajouter(f)  # copie superficielle pour l'affichage "avant"
    s.translater_tout(dx, dy)
    bb1 = s.boite_scene()
    # Le coin inférieur-gauche doit se décaler d'environ (dx,dy)
    assert approx(bb1[0] - bb0[0], dx) and approx(bb1[1] - bb0[1], dy)

    # On affiche : AVANT (bleu pointillé) / APRÈS (orange plein)
    fig, ax = plt.subplots()
    # Avant
    for f in s2.obtenir_figures():
        f.tracer(ax)
        ax.patches[-1].set_linestyle("--")
        ax.patches[-1].set_edgecolor("tab:blue")
    # Après
    for f in s.obtenir_figures():
        f.tracer(ax)
        ax.patches[-1].set_edgecolor("tab:orange")

    # Boîte englobante scène avant/après
    x0,y0,X0,Y0 = bb0
    ax.plot([x0, X0, X0, x0, x0], [y0, y0, Y0, Y0, y0],
            "-", lw=1.0, color="tab:blue", alpha=0.6, label="boîte englobante avant")
    x1,y1,X1,Y1 = bb1
    ax.plot([x1, X1, X1, x1, x1], [y1, y1, Y1, Y1, y1],
            "-", lw=1.0, color="tab:orange", alpha=0.6, label="boîte englobante après")

    ax.set_title("Scène : avant (bleu --) / après translation (orange)")
    cadrer_axes_sur([bb0, bb1], ax)
    ax.legend()
    afficher(fig, "06_scene")

# ------------------------------------------------------------
# Lancement direct
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\nDébut des tests")
    test_01_vecteur2d()
    test_02_point()
    test_03_transformation2d()
    test_04_cercle()
    test_05_rectangle()
    test_06_scene()
    print("\nFin des tests !")
