# src/tp2_optique.py
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
import matplotlib.pyplot as plt

# Adaptation au fait que le module soit dans le sous-dossier "src"
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# On importe les classes de geometrie.py
from src.geometrie import (
    Point, Vecteur2D, Transformation2D,
    Figure, Cercle, Rectangle, Scene
)

# Petites constantes numériques
EPS: float = 1e-9       # décalage après impact pour éviter l'auto-collision
T_MIN: float = 1e-6     # seuil t minimal accepté pour un impact

# ------------------------------------------------------------------------------------
# Utilitaires optiques
# ------------------------------------------------------------------------------------
def reflechir_direction(d: Vecteur2D, n: Vecteur2D) -> Vecteur2D:
    """
    Rôle
    ----
    Calculer la direction réfléchie d'un rayon incident sur une surface
    (réflexion spéculaire parfaite).

    Paramètres
    ----------
    d : Vecteur2D
        Direction incidente (pas forcément unitaire).
    n : Vecteur2D
        Normale SORTANTE à la surface au point d'impact.

    Retour
    ------
    Vecteur2D
        Direction réfléchie, normalisée.

    Notes
    -----
    - Par sécurité, on normalise `n` et la direction finale.
    - Cette fonction ne déplace pas l'origine du rayon.
    """
    n_unit = n.normaliser()
    k = 2.0 * d.produit_scalaire(n_unit)
    return d.soustraire(n_unit.multiplier(k)).normaliser()

# ------------------------------------------------------------------------------------
# Optique : rayon, impact, reflexion
# ------------------------------------------------------------------------------------
@dataclass
class Rayon:
    """
    Rôle
    ----
    Représente un rayon lumineux par son point d'origine et sa direction.

    Attributs
    ---------
    origine : Point
        Point d'émission du rayon.
    direction : Vecteur2D
        Direction de propagation (pas forcément unitaire à la création).

    Remarques
    ---------
    - Utiliser `direction_unitaire()` pour obtenir la direction normalisée,
      sans modifier l'objet.
    """
    origine: Point
    direction: Vecteur2D  # peut ne pas être normalisée à la création

    def direction_unitaire(self) -> Vecteur2D:
        """Retourne une copie normalisée de `direction`."""
        return self.direction.normaliser()


@dataclass
class Impact:
    """
    Rôle
    ----
    Regrouper les informations d'un impact rayon/figure.

    Attributs
    ---------
    figure : Figure
        Figure touchée.
    t : float
        Paramètre du rayon tel que P = O + t * d, avec t >= 0.
    point : Point
        Point d'impact.
    normale : Vecteur2D
        Normale SORTANTE au point d'impact (à traiter comme unitaire).
    """
    figure: Figure
    t: float
    point: Point
    normale: Vecteur2D

# -----------------------------
# Extensions optiques des figures
# -----------------------------
class CercleOptique(Cercle):
    def intersection_rayon(self, o: Point, d: Vecteur2D) -> tuple[float, Point, Vecteur2D] | None:
        """
        Rôle
        ----
        Tester l'intersection entre le rayon paramétré `o + t d` (t >= 0) et ce cercle,
        et retourner l'impact le plus proche s'il existe.

        Paramètres
        ----------
        o : Point
            Origine du rayon.
        d : Vecteur2D
            Direction du rayon (peut ne pas être unitaire).

        Retour
        ------
        (t, P, n) ou None
          - t : distance paramétrique >= 0
          - P : point d'impact (Point)
          - n : normale sortante unitaire (Vecteur2D)
          - None si aucune intersection valable
        """
        oc = o.vers_vecteur().soustraire(self.obtenir_centre().vers_vecteur())
        b = 2.0 * d.produit_scalaire(oc)
        c = oc.produit_scalaire(oc) - self.obtenir_rayon() * self.obtenir_rayon()
        disc = b*b - 4.0*c
        if disc < 0: 
            return None
        s = sqrt(disc)
        t1 = (-b - s) / 2.0
        t2 = (-b + s) / 2.0
        t = None
        for cand in (t1, t2):
            if cand >= 0 and (t is None or cand < t):
                t = cand
        if t is None:
            return None
        hit = Vecteur2D(o.obtenir_x(), o.obtenir_y()).ajouter(d.multiplier(t))
        P = Point(hit.x, hit.y)
        n = hit.soustraire(self.obtenir_centre().vers_vecteur()).normaliser()
        return (t, P, n)


class RectangleOptique(Rectangle):
    def intersection_rayon(self, o: Point, d: Vecteur2D) -> tuple[float, Point, Vecteur2D] | None:
        """
        Rôle
        ----
        Tester l'intersection entre le rayon `o + t d` (t >= 0) et ce rectangle
        aligné avec les axes (méthode des bandes).

        Paramètres
        ----------
        o : Point
            Origine du rayon.
        d : Vecteur2D
            Direction du rayon (peut ne pas être unitaire).

        Retour
        ------
        (t, P, n) ou None
          - t : distance paramétrique >= 0
          - P : point d'impact (Point)
          - n : normale sortante unitaire parmi {(-1,0),(1,0),(0,-1),(0,1)}
          - None si aucune intersection valable
        """
        xmin, ymin, xmax, ymax = self.boite_englobante()
        ox, oy = o.obtenir_x(), o.obtenir_y()
        dx, dy = d.x, d.y

        def inv(x): return (1.0 / x) if abs(x) > 1e-15 else float("inf")

        inv_dx, inv_dy = inv(dx), inv(dy)
        t1 = (xmin - ox) * inv_dx
        t2 = (xmax - ox) * inv_dx
        t3 = (ymin - oy) * inv_dy
        t4 = (ymax - oy) * inv_dy

        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))
        if tmax < 0 or tmin > tmax:  # derrière ou pas d'intersection
            return None
        t = tmin if tmin >= 0 else tmax
        if t < 0:
            return None

        hit = Vecteur2D(ox, oy).ajouter(d.multiplier(t))
        hx, hy = hit.x, hit.y

        eps = 1e-10
        if abs(hx - xmin) < eps: n = Vecteur2D(-1, 0)
        elif abs(hx - xmax) < eps: n = Vecteur2D(1, 0)
        elif abs(hy - ymin) < eps: n = Vecteur2D(0, -1)
        elif abs(hy - ymax) < eps: n = Vecteur2D(0, 1)
        else:
            candidats = [(abs(hx - xmin), Vecteur2D(-1, 0)),
                         (abs(hx - xmax), Vecteur2D(1, 0)),
                         (abs(hy - ymin), Vecteur2D(0, -1)),
                         (abs(hy - ymax), Vecteur2D(0, 1))]
            n = min(candidats, key=lambda p: p[0])[1]
        return (t, Point(hx, hy), n)  # n est déjà unitaire


# -----------------------------
# Scene optique (séparée de la Scene TP1)
# -----------------------------
class SceneOptique(Scene):
    def premier_impact(self, rayon: Rayon, t_min: float = T_MIN, t_max: float = 1e9) -> Impact | None:
        """
        Rôle
        ----
        Trouver, parmi les figures de la scène, l'impact valide le plus proche
        du rayon donné.

        Paramètres
        ----------
        rayon : Rayon
            Rayon à tester.
        t_min : float
            Seuil inférieur pour ignorer les auto-collisions/tangences.
        t_max : float
            Seuil supérieur (borne haute).

        Retour
        ------
        Impact | None
            Impact le plus proche dans [t_min, t_max] ou None si aucune intersection.
        """
        o, d = rayon.origine, rayon.direction_unitaire()
        best = None
        for f in self.obtenir_figures():
            # On ne considère que les figures ayant la méthode intersection_rayon
            meth = getattr(f, "intersection_rayon", None)
            if meth is None:  # figure TP1 brute (sans optique)
                continue
            res = meth(o, d)
            if res is None:
                continue
            t, P, n = res
            if t_min <= t <= t_max and (best is None or t < best.t):
                best = Impact(f, t, P, n)
        return best

    def suivre_rayon(self, rayon: Rayon, nb_rebonds_max: int = 3) -> list[Point]:
        """
        Rôle
        ----
        Suivre un rayon avec réflexions spéculaires parfaites pendant au plus
        `nb_rebonds_max` rebonds. Retourner la polyligne des points.

        Retour
        ------
        list[Point]
            [origine, impact1, impact2, ..., fin]
        """
        pts = [rayon.origine]
        cur_o = rayon.origine
        cur_d = rayon.direction_unitaire()
        for _ in range(nb_rebonds_max):
            hit = self.premier_impact(Rayon(cur_o, cur_d))
            if hit is None:
                # prolonge pour visualiser la sortie
                far = Point(cur_o.obtenir_x() + 1000 * cur_d.x,
                            cur_o.obtenir_y() + 1000 * cur_d.y)
                pts.append(far)
                break
            pts.append(hit.point)
            # nouvelle origine : léger décalage pour éviter l'auto-collision
            cur_o = Point(hit.point.obtenir_x() + EPS * hit.normale.x,
                          hit.point.obtenir_y() + EPS * hit.normale.y)
            cur_d = reflechir_direction(cur_d, hit.normale)  # déjà normalisée par la fonction
        return pts

    @staticmethod
    def scene_exemple() -> "SceneOptique":
        sc = SceneOptique()
        # boite murale (4 rectangles minces)
        sc.ajouter(RectangleOptique(Point(-4, -2.5), 8, 0.1))   # bas
        sc.ajouter(RectangleOptique(Point(-4, 2.4), 8, 0.1))    # haut
        sc.ajouter(RectangleOptique(Point(-4, -2.5), 0.1, 5))   # gauche
        sc.ajouter(RectangleOptique(Point(3.9, -2.5), 0.1, 5))  # droite
        # obstacle circulaire
        sc.ajouter(CercleOptique(Point(0.5, 0.0), 1.0))
        return sc

    def tracer_rayons(self, polylignes: list[list[Point]] | None = None, montrer_boite: bool = True) -> None:
        """
        Affiche la scène et, si fourni, les trajectoires (polylignes) des rayons.

        Conventions visuelles :
        - polylignes en tirets fins,
        - départ ★, impacts •, sortie X.
        """
        fig, ax = plt.subplots()

        # Tracer les figures
        for f in self.obtenir_figures():
            f.tracer(ax)

        # Option : bbox globale
        if montrer_boite and self.obtenir_figures():
            x0, y0, x1, y1 = self.boite_scene()
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                    "k:", lw=1)

        # Tracer les polylignes (rayons)
        if polylignes:
            for poly in polylignes:
                xs = [p.obtenir_x() for p in poly]
                ys = [p.obtenir_y() for p in poly]
                # ligne en tirets fins
                ax.plot(xs, ys, "--", lw=1.0)
                # départ (★), impacts intermédiaires (•), fin (X)
                if len(xs) >= 1:
                    ax.scatter(xs[0], ys[0], marker="*", s=120, color="tab:blue", edgecolors="black")
                if len(xs) >= 2:
                    for k in range(1, len(xs)-1):
                        ax.scatter(xs[k], ys[k], s=30, color="tab:blue", edgecolors="black", linewidths=0.4)
                    ax.scatter(xs[-1], ys[-1], marker="X", s=80, color="tab:blue", edgecolors="black")

        # Cadre général
        if self.obtenir_figures():
            x0, y0, x1, y1 = self.boite_scene()
            pad = 0.2 * max(1.0, (x1 - x0) + (y1 - y0))
            ax.set_xlim(x0 - pad, x1 + pad)
            ax.set_ylim(y0 - pad, y1 + pad)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":")
        ax.set_title("Scène optique : figures et rayons")
        plt.show()
