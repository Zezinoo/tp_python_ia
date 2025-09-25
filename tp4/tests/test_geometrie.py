# tests/test_geometrie.py

import os
from math import isclose, pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.geometrie import (
    Point, Vecteur2D, Transformation2D,
    Cercle, Rectangle, Scene
)

SHOW = True
SAVE = False
OUT_DIR = "tests/_sortie_tp1"
if SAVE and not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

def approx(a, b, rel=1e-7, abs_=1e-9): return isclose(a, b, rel_tol=rel, abs_tol=abs_)
def banniere(t): print("\n"+"="*len(t)); print(t); print("="*len(t))
def afficher(fig, name):
    if SAVE: fig.savefig(os.path.join(OUT_DIR, f"{name}.svg"), dpi=120, bbox_inches="tight")
    if SHOW: plt.show()
    else: plt.close(fig)

def cadrer(ax, bbs, pad_ratio=0.2):
    xs = [b[0] for b in bbs] + [b[2] for b in bbs]
    ys = [b[1] for b in bbs] + [b[3] for b in bbs]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    pad = pad_ratio * max(1.0, (x1-x0)+(y1-y0))
    ax.set_xlim(x0-pad, x1+pad); ax.set_ylim(y0-pad, y1+pad)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, linestyle=":", alpha=0.5)

def test_01_vecteur2d():
    banniere("01) Vecteur2D — opérations de base")
    v = Vecteur2D(3,4); assert approx(v.norme(), 5.0)
    vn = v.normaliser(); assert approx(vn.norme(), 1.0)
    assert approx(v.ajouter(Vecteur2D(1,-2)).x, 4)
    assert approx(v.soustraire(Vecteur2D(2,1)).y, 3)
    assert approx(v.multiplier(2).x, 6)
    assert approx(v.produit_scalaire(Vecteur2D(-4,1)), -8)

    fig, ax = plt.subplots()
    ax.quiver(0,0,v.x,v.y,angles='xy',scale_units='xy',scale=1,color='tab:blue',width=0.007,label="v")
    ax.quiver(0,0,vn.x,vn.y,angles='xy',scale_units='xy',scale=1,color='tab:orange',width=0.007,label="normalise(v)")
    L = max(1.0, v.norme())*1.3
    ax.set_xlim(-0.2, L); ax.set_ylim(-0.2, L)
    ax.legend(); ax.set_title("Vecteur2D : v (bleu) / normaliser(v) (orange)")
    afficher(fig, "01_vecteur2d")

def test_02_point():
    banniere("02) Point — translation & conversion")
    p0 = Point(1,2); p = Point(1,2); p.translater(3,0)
    assert approx(p.obtenir_x(), 4) and approx(p.obtenir_y(), 2)
    fig, ax = plt.subplots()
    ax.plot(p0.obtenir_x(), p0.obtenir_y(), "o", color="tab:blue", label="avant")
    ax.plot(p.obtenir_x(),  p.obtenir_y(),  "o", color="tab:orange", label="Après")
    cadrer(ax, [(min(1,4),2, max(1,4),2)])
    ax.legend(); ax.set_title("Point : translation")
    afficher(fig, "02_point")

def test_03_transformation2d():
    banniere("03) Transformation2D — rotation/translation/compose/mise à l'échelle")
    p = Point(1,0)
    Trot = Transformation2D.rotation(pi/2)
    p2 = Trot.appliquer_point(p); assert approx(p2.obtenir_x(), 0.0) and approx(p2.obtenir_y(), 1.0)
    Ttr = Transformation2D.translation(2,-1)
    p3 = Ttr.appliquer_point(p2); assert approx(p3.obtenir_x(), 2.0) and approx(p3.obtenir_y(), 0.0)
    T = Ttr.composer(Trot)
    p4 = T.appliquer_point(Point(1,0)); assert approx(p4.obtenir_x(), 2.0) and approx(p4.obtenir_y(), 0.0)
    S = Transformation2D.mise_echelle(2,3)
    p5 = S.appliquer_point(Point(1,1)); assert approx(p5.obtenir_x(), 2.0) and approx(p5.obtenir_y(), 3.0)

    r = Rectangle(Point(0,0),2,1); r2 = r.transformer(T)
    fig, ax = plt.subplots()
    rect1 = patches.Rectangle((r.obtenir_coin().obtenir_x(), r.obtenir_coin().obtenir_y()),
                              r.obtenir_largeur(), r.obtenir_hauteur(),
                              fill=False, edgecolor="tab:blue", linewidth=2, label="Avant")
    ax.add_patch(rect1)
    bb2 = r2.boite_englobante()
    rect2 = patches.Rectangle((bb2[0], bb2[1]), bb2[2]-bb2[0], bb2[3]-bb2[1],
                              fill=False, edgecolor="tab:orange", linewidth=2, label="Après")
    ax.add_patch(rect2)
    cadrer(ax, [r.boite_englobante(), r2.boite_englobante()])
    ax.legend(); ax.set_title("Transformation2D : Avant (bleu) / Après (orange)")
    afficher(fig, "03_transformation")

def test_04_cercle():
    banniere("04) Cercle — métriques, boite englobante, contenance")
    c = Cercle(Point(0,0), 2)
    assert approx(c.aire(), pi*4) and approx(c.perimetre(), 4*pi)
    assert c.contient(Point(0.5,0.5)) and not c.contient(Point(2.1,0.0))
    fig, ax = plt.subplots()
    c.tracer(ax)
    ax.plot([0.5,2.1],[0.5,0.0],"o",color="tab:red"); ax.text(0.5,0.5,"in"); ax.text(2.1,0.0,"out")
    cadrer(ax, [c.boite_englobante()])
    ax.set_title("Cercle : boite englobante & contenance")
    afficher(fig, "04_cercle")

def test_05_rectangle():
    banniere("05) Rectangle — métriques, boite englobante, contenance")
    r = Rectangle(Point(-1,-0.5), 2, 1)
    assert approx(r.aire(), 2.0) and approx(r.perimetre(), 6.0)
    assert r.contient(Point(0,0)) and not r.contient(Point(1.01,0.5))
    fig, ax = plt.subplots()
    r.tracer(ax)
    x0,y0,x1,y1 = r.boite_englobante()
    ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],"C2--")
    ax.plot([0,1.01],[0,0.5],"o",color="tab:red")
    cadrer(ax, [r.boite_englobante()])
    ax.set_title("Rectangle : boite englobante & contenance")
    afficher(fig, "05_rectangle")

def test_06_scene():
    banniere("06) Scene — ajout, boite englobante, translation globale")
    s = Scene(); s.ajouter(Cercle(Point(0,0),1)); s.ajouter(Rectangle(Point(-1,-0.5),2,1))
    bb0 = s.boite_scene()
    s_avant = Scene(); [s_avant.ajouter(f) for f in s.obtenir_figures()]
    s.translater_tout(1.0, 0.5)
    bb1 = s.boite_scene()
    assert approx(bb1[0]-bb0[0], 1.0) and approx(bb1[1]-bb0[1], 0.5)

    fig, ax = plt.subplots()
    # AVANT (bleu pointille)
    for f in s_avant.obtenir_figures():
        b = f.boite_englobante()
        ax.plot([b[0],b[2],b[2],b[0],b[0]],[b[1],b[1],b[3],b[3],b[1]], "tab:blue", linestyle="--")
    ax.plot([bb0[0],bb0[2],bb0[2],bb0[0],bb0[0]],[bb0[1],bb0[1],bb0[3],bb0[3],bb0[1]],
            "tab:blue", linestyle="--", label="Avant")
    # APRES (orange plein)
    for f in s.obtenir_figures():
        b = f.boite_englobante()
        ax.plot([b[0],b[2],b[2],b[0],b[0]],[b[1],b[1],b[3],b[3],b[1]], "tab:orange")
    ax.plot([bb1[0],bb1[2],bb1[2],bb1[0],bb1[0]],[bb1[1],bb1[1],bb1[3],bb1[3],bb1[1]],
            "tab:orange", label="Après")
    # fleche de translation
    cx0, cy0 = (bb0[0]+bb0[2])/2, (bb0[1]+bb0[3])/2
    cx1, cy1 = (bb1[0]+bb1[2])/2, (bb1[1]+bb1[3])/2
    ax.annotate("translation", xy=(cx1,cy1), xytext=(cx0,cy0),
                arrowprops=dict(arrowstyle="->", lw=2, color="tab:orange"),
                color="tab:orange")
    cadrer(ax, [bb0, bb1]); ax.legend(); ax.set_title("Scene — Avant / Après")
    afficher(fig, "06_scene")

if __name__ == "__main__":
    print("\nDébut des tests")
    test_01_vecteur2d()
    test_02_point()
    test_03_transformation2d()
    test_04_cercle()
    test_05_rectangle()
    test_06_scene()
    print("\nFin des tests !")