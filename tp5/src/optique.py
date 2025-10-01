# src/optique.py
from __future__ import annotations
from dataclasses import dataclass
from math import isclose
import matplotlib.pyplot as plt
import numpy as np

# Adaptation au fait que le module soit dans le sous-dossier "src"
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# On importe les classes du TP1
from src.geometrie import (
    Point, Vecteur2D, Transformation2D,   # Transformation2D importée au cas où
    Figure, Cercle, Rectangle, Scene
)

# ---------------------------------------------------------------------
# Constantes numériques
# ---------------------------------------------------------------------
EPS: float = 1e-9    # décalage après impact pour éviter l'auto-collision
T_MIN: float = 1e-6  # seuil t minimal accepté pour un impact


# ---------------------------------------------------------------------
# Utilitaires optiques
# ---------------------------------------------------------------------
def reflechir_direction(d: Vecteur2D, n: Vecteur2D) -> Vecteur2D:
    """
    Rôle
    ----
    Calculer la direction réfléchie d'un rayon incident sur une surface
    en appliquant la loi de la réflexion (réflexion spéculaire parfaite).

    Paramètres
    ----------
    d : Vecteur2D
        Direction incidente du rayon (pas forcément unitaire).
    n : Vecteur2D
        Normale sortante à la surface au point d'impact (à traiter comme unitaire).

    Retour
    ------
    Vecteur2D
        Nouvelle direction du rayon après réflexion (normalisée).

    Indications
    -----------
    - Normaliser `n` par précaution avant usage.
    - Appliquer la formule de réflexion :
      d' = d - 2 * (d·n) * n
    - Normaliser le vecteur résultat avant de le retourner.
    """
    n = n.normaliser()
    reflected = d.soustraire( n.multiplier(2 * (d.produit_scalaire(n))))
    reflected = reflected.normaliser()

    return reflected

def approx(a, b, rel=1e-7, abs_=1e-9): 
    return isclose(a, b, rel_tol=rel, abs_tol=abs_)

# ---------------------------------------------------------------------
# Optique : rayon, impact (conteneurs de données)
# ---------------------------------------------------------------------
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
        Direction de propagation (peut ne pas être unitaire à la création).

    Remarques
    ---------
    - Utiliser `direction_unitaire()` pour obtenir la direction normalisée,
      sans modifier l'objet.
    """
    origine: Point
    direction: Vecteur2D

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
        Normale sortante au point d'impact (traitée comme unitaire).
    """
    figure: Figure
    t: float
    point: Point
    normale: Vecteur2D


# ---------------------------------------------------------------------
# Extensions optiques des figures
# ---------------------------------------------------------------------
class CercleOptique(Cercle):
    def intersection_rayon(self, o: Point, d: Vecteur2D) -> tuple[float, Point, Vecteur2D] | None:
        """
        Rôle
        ----
        Tester l'intersection entre le rayon paramétré "o + t d" (t >= 0) et ce cercle,
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

        Indications
        -----------
        - Rappel (fiche) : résoudre |o + t d - C|^2 = R^2 (équation quadratique en t).
        - Conserver la plus petite solution t >= 0 (si elle existe).
        - Construire P à partir de o, d et t ; calculer la normale sortante en P.
        """
        centre = self.obtenir_centre()
        oc = Vecteur2D(o.obtenir_x() - centre.obtenir_x() , o.obtenir_y() -centre.obtenir_y() )
        d_dot_d = d.produit_scalaire(d)
        oc_dot_d = oc.produit_scalaire(d)
        oc_dot_oc = oc.produit_scalaire(oc)
        r_squared = self.obtenir_rayon()**2

        # Resoudre la equation quadratique
        a = d_dot_d
        b = 2 * oc_dot_d
        c = oc_dot_oc - r_squared
        
        delta = b**2 - 4*a*c

        t_1 = (-b + np.sqrt(delta))/(2*a) 
        t_2 = (-b - np.sqrt(delta))/(2*a)

        t = min(filter(lambda x : x > 0 , [t_1 , t_2]),default = -1)

        if t < 0 :
            return None , None , None
        
        p_x = o.obtenir_x() + t*d.x
        p_y = o.obtenir_y() + t*d.y

        p = Point(p_x , p_y)

        n_x = p_x - centre.obtenir_x()
        n_y = p_y - centre.obtenir_y()

        n = Vecteur2D(n_x , n_y).normaliser()

        return t , p , n


class RectangleOptique(Rectangle):
    def intersection_rayon(self, o: Point, d: Vecteur2D) -> tuple[float, Point, Vecteur2D] | None:
        """
        Rôle
        ----
        Tester l'intersection entre le rayon "o + t d" (t >= 0) et ce rectangle
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

        Indications
        -----------
        - Appliquer pas-à-pas la *méthode des bandes* (pseudo-code fourni dans le sujet).
        - Déterminer la face touchée pour fixer la normale sortante.
        """
        xmin , ymin , xmax , ymax = self.boite_englobante()
        ox = o.obtenir_x()
        oy = o.obtenir_y()

        dx = d.x
        dy = d.y

        t1 = (xmin - ox) / dx
        t2 = (xmax - ox) / dx
        txmin = min(t1, t2)
        txmax = max(t1, t2)

        t3 = (ymin - oy) / dy
        t4 = (ymax - oy) / dy
        tymin = min(t3, t4)
        tymax = max(t3, t4)

        t_enter = max(txmin, tymin)
        t_sortie = min(txmax, tymax)

        if t_enter <= t_sortie and t_sortie >= 0:
            t = t_enter
        else:
            return None , None , None

        p_x = o.obtenir_x() + t*d.x
        p_y = o.obtenir_y() + t*d.y

        p = Point(p_x , p_y)
        
        normal_direction = {"gauche" : (-1,0), 
                            "droite" : (1,0),
                            "bas" : (0,-1), 
                            "haut" : (0,1) }
        face = ""

        if isclose(p_x  , xmin):
            face = "gauche"
        elif isclose(p_x , xmax):
            face = "droite"
        else:
            if isclose(p_y , ymin):
                face = "bas"
            elif isclose(p_y , ymax):
                face = "haut"
            else:
                raise ValueError("something wrong :(")
        r = normal_direction[face]

        n = Vecteur2D(r[0] , r[1])
            
        return t,p,n

# ---------------------------------------------------------------------
# Scène optique (séparée de la Scene TP1)
# ---------------------------------------------------------------------
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

        Indications
        -----------
        - Parcourir les figures de la scène (celles qui exposent "intersection_rayon()").
        - Conserver l'impact avec le plus petit t dans l'intervalle [t_min, t_max].
        """
        dict_impact = {"t" : None , "n" : None , "p" : None , "f" : None}
        o = rayon.origine
        d = rayon.direction


        for f in self.obtenir_figures():
            t , p , n = f.intersection_rayon(o , d)
            if t is None:
                continue
            elif t < t_min or t > t_max:
                continue
            elif dict_impact["t"] is None :
                dict_impact["t"] = t
                dict_impact["n"] = n
                dict_impact["p"] = p
                dict_impact["f"] = f
            elif t < dict_impact["t"]:
                dict_impact["t"] = t
                dict_impact["n"] = n
                dict_impact["p"] = p
                dict_impact["f"] = f
        
            if dict_impact["t"] < 0 :
                dict_impact["t"] = None
                dict_impact["n"] = None
                dict_impact["p"] = None
                dict_impact["f"] = None

        return Impact(dict_impact["f"] , dict_impact["t"] , dict_impact["p"] , dict_impact["n"])

    def suivre_rayon(self, rayon: Rayon, nb_rebonds_max: int = 3) -> list[Point]:
        """
        Rôle
        ----
        Suivre un rayon avec réflexions spéculaires parfaites pendant au plus
        nb_rebonds_max rebonds. Retourner la polyligne des points visités.

        Retour
        ------
        list[Point]
            [origine, impact1, impact2, ..., fin]

        Indications
        -----------
        - Utiliser premier_impact à chaque étape.
        - Après un impact, appliquer reflechir_direction() pour mettre à jour la direction.
        - Repartir de "P + EPS * n" pour éviter l'auto-collision numérique.
        - Si aucun impact n'est trouvé : prolonger "loin = origine + 1000 * direction" et terminer.
        """
        points = [rayon.origine]

        for i in range(nb_rebonds_max):
            current_impact = self.premier_impact(rayon)
            if current_impact.t is None:
                o = rayon.origine
                travel = rayon.direction_unitaire().multiplier(1000)
                fin = Point(o.obtenir_x() + travel.x , o.obtenir_y()  + travel.y)
                points.append(fin)
                break
            d = rayon.direction
            n = current_impact.normale
            p = current_impact.point
            reflected_direction = reflechir_direction(d , n)
            new_p = Point(p.obtenir_x() + EPS , p.obtenir_y() + EPS)
            points.append(new_p)
            reflected_ray = Rayon(new_p , reflected_direction)
            rayon = reflected_ray



        return points

    @staticmethod
    def scene_exemple() -> "SceneOptique":
        """
        Rôle
        ----
        Construire une petite scène de test :
        - une "boîte" rectangulaire (4 murs minces),
        - un obstacle circulaire au centre.
        """
        sc = SceneOptique()
        # Boîte (4 rectangles minces)
        sc.ajouter(RectangleOptique(Point(-4, -2.5), 8, 0.1))     # bas
        sc.ajouter(RectangleOptique(Point(-4,  2.4), 8, 0.1))     # haut
        sc.ajouter(RectangleOptique(Point(-4, -2.5), 0.1, 5.0))   # gauche
        sc.ajouter(RectangleOptique(Point( 3.9, -2.5), 0.1, 5.0)) # droite
        # Obstacle circulaire
        sc.ajouter(CercleOptique(Point(0.5, 0.0), 1.0))
        return sc

    def tracer_rayons(self, polylignes: list[list[Point]] | None = None, montrer_boite: bool = True) -> None:
        """
        Affiche la scène et, si fournies, les trajectoires (polylignes) des rayons.

        Conventions :
        - polylignes en tirets fins,
        - départ : "*", impacts : ".", sortie : "X".
        """
        fig, ax = plt.subplots()

        # Tracer les figures
        for f in self.obtenir_figures():
            f.tracer(ax)

        # Option : boîte englobante globale
        if montrer_boite and self.obtenir_figures():
            x0, y0, x1, y1 = self.boite_scene()
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k:", lw=1)

        # Tracer les polylignes (rayons)
        if polylignes:
            for poly in polylignes:
                xs = [p.obtenir_x() for p in poly]
                ys = [p.obtenir_y() for p in poly]
                # ligne en tirets fins
                ax.plot(xs, ys, "--", lw=1.0)
                # départ (*), impacts intermédiaires (.), fin (X)
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
