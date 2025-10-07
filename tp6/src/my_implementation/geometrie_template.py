# src/geometrie.py
"""
=======================================
     TP POO n°1 — Géométrie 2D
=======================================

Ce module définit les classes à implémenter pour le TP :
- Vecteur2D : opérations vectorielles de base
- Point : position dans le plan
- Transformation2D : transformations affines (translation, rotation, mise à l'échelle)
- Figure (abstraite) : contrat commun
- Cercle, Rectangle : figures concrètes
- Scene : agrégation de figures et visualisation

Consignes :
- Respecter les signatures ci-dessous.
- Lire chaque docstring avant de coder : rôle, paramètres, retour, exceptions.
- Utiliser une petite tolérance numérique EPS pour les comparaisons réelles.
- Les méthodes `transformer` retournent une *nouvelle* figure (ne pas muter l'objet courant).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt, pi, cos, sin
from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches

EPS = 1e-12  # tolérance numérique pour les comparaisons (bords inclus, etc.)


# ---------------------------------------------------------------------
# 1) Vecteur2D
# ---------------------------------------------------------------------
@dataclass
class Vecteur2D:
    """
    Représente un vecteur 2D (direction, déplacement).
    Attributs
    ---------
    x, y : float
        Composantes cartésiennes du vecteur.
    """
    x: float
    y: float

    def ajouter(self, v: "Vecteur2D") -> "Vecteur2D":
        """
        Calcule la somme de deux vecteurs (self + v).

        Paramètres
        ----------
        v : Vecteur2D
            Vecteur à additionner.

        Retour
        ------
        Nouveau Vecteur2D
        """
        raise NotImplementedError("A COMPLETER")

    def soustraire(self, v: "Vecteur2D") -> "Vecteur2D":
        """
        Calcule la différence de deux vecteurs (self - v).

        Paramètres
        ----------
        v : Vecteur2D
            Vecteur à soustraire.

        Retour
        ------
        Nouveau Vecteur2D
        """
        raise NotImplementedError("A COMPLETER")

    def multiplier(self, k: float) -> "Vecteur2D":
        """
        Multiplie le vecteur par un scalaire.

        Paramètres
        ----------
        k : float
            Facteur d'échelle.

        Retour
        ------
        Nouveau Vecteur2D
        """
        raise NotImplementedError("A COMPLETER")

    def produit_scalaire(self, v: "Vecteur2D") -> float:
        """
        Calcule le produit scalaire (dot product) self·v.

        Paramètres
        ----------
        v : Vecteur2D

        Retour
        ------
        float
        """
        raise NotImplementedError("A COMPLETER")

    def norme(self) -> float:
        """
        Calcule la norme euclidienne du vecteur.

        Retour
        ------
        float
        """
        raise NotImplementedError("A COMPLETER")

    def normaliser(self) -> "Vecteur2D":
        """
        Retourne un vecteur unitaire de même direction.

        Retour
        ------
        Vecteur2D
            Vecteur normalisé. Si la norme vaut 0, retourner (0,0).
        """
        raise NotImplementedError("A COMPLETER")

    def perpendiculaire_gauche(self) -> "Vecteur2D":
        """
        Retourne le vecteur perpendiculaire gauche (rotation +90°).

        Retour
        ------
        Vecteur2D
        """
        raise NotImplementedError("A COMPLETER")


# ---------------------------------------------------------------------
# 2) Point
# ---------------------------------------------------------------------
class Point:
    """
    Représente une position dans le plan.
    Attributs privés
    ----------------
    _x, _y : float
        Coordonnées cartésiennes.
    """

    def __init__(self, x: float, y: float):
        """
        Crée un point.

        Paramètres
        ----------
        x, y : float
            Coordonnées du point.
        """
        raise NotImplementedError("A COMPLETER")

    def obtenir_x(self) -> float:
        """
        Lit l'abscisse du point.

        Retour
        ------
        float
            Valeur de x.
        """
        raise NotImplementedError("A COMPLETER")

    def obtenir_y(self) -> float:
        """
        Lit l'ordonnée du point.

        Retour
        ------
        float
            Valeur de y.
        """
        raise NotImplementedError("A COMPLETER")

    def definir_x(self, v: float) -> None:
        """
        Modifie l'abscisse du point.

        Paramètres
        ----------
        v : float
            Nouvelle valeur de x.
        """
        raise NotImplementedError("A COMPLETER")

    def definir_y(self, v: float) -> None:
        """
        Modifie l'ordonnée du point.

        Paramètres
        ----------
        v : float
            Nouvelle valeur de y.
        """
        raise NotImplementedError("A COMPLETER")

    def translater(self, dx: float, dy: float) -> None:
        """
        Translate le point par (dx, dy) (modifie *en place*).

        Paramètres
        ----------
        dx, dy : float
            Incréments à ajouter à (x, y).
        """
        raise NotImplementedError("A COMPLETER")

    def vers_vecteur(self) -> Vecteur2D:
        """
        Convertit le point en vecteur (x, y).

        Retour
        ------
        Vecteur2D
        """
        raise NotImplementedError("A COMPLETER")

    def __repr__(self) -> str:
        """
        Représentation lisible (utile pour le débogage).

        Retour
        ------
        str
            Exemple : "Point(x=1.000, y=2.000)".
        """
        raise NotImplementedError("A COMPLETER")


# ---------------------------------------------------------------------
# 3) Transformation2D
# ---------------------------------------------------------------------
class Transformation2D:
    """
    Représente une transformation affine 2D :
    matrice 2x2 [[a,b],[c,d]] et translation (tx, ty).
    Application à un point P=(x,y) :
        P' = M*P + t
    """

    def __init__(self, a: float, b: float, c: float, d: float, tx: float, ty: float):
        """
        Crée une transformation affine.

        Paramètres
        ----------
        a, b, c, d : float
            Coefficients de la matrice 2x2.
        tx, ty : float
            Composantes de la translation.
        """
        raise NotImplementedError("A COMPLETER")

    @staticmethod
    def identite() -> "Transformation2D":
        """
        Fabrique la transformation identité.

        Retour
        ------
        Transformation2D
        """
        raise NotImplementedError("A COMPLETER")

    @staticmethod
    def translation(dx: float, dy: float) -> "Transformation2D":
        """
        Fabrique une translation.

        Paramètres
        ----------
        dx, dy : float
            Translation à appliquer.

        Retour
        ------
        Transformation2D
        """
        raise NotImplementedError("A COMPLETER")

    @staticmethod
    def rotation(theta_rad: float) -> "Transformation2D":
        """
        Fabrique une rotation d'angle theta (en radians).

        Paramètres
        ----------
        theta_rad : float
            Angle en radians.

        Retour
        ------
        Transformation2D
        """
        raise NotImplementedError("A COMPLETER")

    @staticmethod
    def mise_echelle(sx: float, sy: float) -> "Transformation2D":
        """
        Fabrique une mise à l'échelle (anisotrope possible).

        Paramètres
        ----------
        sx, sy : float
            Facteurs d'échelle en x et y.

        Retour
        ------
        Transformation2D
        """
        raise NotImplementedError("A COMPLETER")

    def appliquer_point(self, p: Point) -> Point:
        """
        Applique la transformation au point p et retourne un *nouveau* point.
        (x’, y’) = (a.x + b.y + tx, ; c.x + d.y + ty)

        Paramètres
        ----------
        p : Point

        Retour
        ------
        Point (Le point transformé)
        """
        raise NotImplementedError("A COMPLETER")

    def composer(self, autre: "Transformation2D") -> "Transformation2D":
        """
        Compose deux transformations (self o autre).

        Convention
        ----------
        Le résultat applique d'abord 'autre', puis 'self' :
            P' = self(autre(P))

        Paramètres
        ----------
        autre : Transformation2D

        Retour
        ------
        Transformation2D
        """
        raise NotImplementedError("A COMPLETER")


# ---------------------------------------------------------------------
# 4) Figure (abstraite)
# ---------------------------------------------------------------------
class Figure(ABC):
    """
    Contrat commun à toutes les figures.
    Les implémentations concrètes doivent respecter ces signatures.
    """

    @abstractmethod
    def aire(self) -> float:
        """
        Aire géométrique de la figure.

        Retour
        ------
        float
            Aire (>= 0).
        """
        raise NotImplementedError("A COMPLETER")

    @abstractmethod
    def perimetre(self) -> float:
        """
        Périmètre (longueur du contour).

        Retour
        ------
        float
            Périmètre (>= 0).
        """
        raise NotImplementedError("A COMPLETER")

    @abstractmethod
    def boite_englobante(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante axis-aligned de la figure.

        Retour
        ------
        (xmin, ymin, xmax, ymax) : tuple[float,float,float,float]
            Avec xmin ≤ xmax, ymin ≤ ymax.
        """
        raise NotImplementedError("A COMPLETER")

    @abstractmethod
    def contient(self, p: Point) -> bool:
        """
        Teste si un point appartient à la figure (bords inclus).

        Paramètres
        ----------
        p : Point

        Retour
        ------
        bool
        """
        raise NotImplementedError("A COMPLETER")

    @abstractmethod
    def transformer(self, T: Transformation2D) -> "Figure":
        """
        Retourne une *nouvelle* figure transformée par T.

        Paramètres
        ----------
        T : Transformation2D

        Retour
        ------
        Figure
            Nouvelle instance (ne pas modifier l'original).
        """
        raise NotImplementedError("A COMPLETER")

    @abstractmethod
    def tracer(self, ax) -> None:
        """
        Dessine la figure dans un Axes matplotlib.

        Paramètres
        ----------
        ax : matplotlib.axes.Axes
            Axes cible. Utiliser patches.Circle / patches.Rectangle, fill=False.
        """
        raise NotImplementedError("A COMPLETER")


# ---------------------------------------------------------------------
# 5) Cercle
# ---------------------------------------------------------------------
class Cercle(Figure):
    """
    Cercle défini par son centre et son rayon (> 0).
    Attributs privés
    ----------------
    _centre : Point
    _rayon  : float
    """

    def __init__(self, centre: Point, rayon: float):
        """
        Construit un cercle.

        Paramètres
        ----------
        centre : Point
        rayon  : float
            Doit être strictement positif.

        Exceptions
        ----------
        ValueError : si rayon <= 0.
        """
        raise NotImplementedError("A COMPLETER")

    def obtenir_centre(self) -> Point:
        """
        Retourne le centre du cercle.

        Retour
        ------
        Point
        """
        raise NotImplementedError("A COMPLETER")

    def obtenir_rayon(self) -> float:
        """
        Retourne le rayon du cercle.

        Retour
        ------
        float
        """
        raise NotImplementedError("A COMPLETER")

    def definir_rayon(self, r: float) -> None:
        """
        Modifie le rayon.

        Paramètres
        ----------
        r : float
            Doit être > 0.

        Exceptions
        ----------
        ValueError : si r <= 0.
        """
        raise NotImplementedError("A COMPLETER")

    # --- Figure ---

    def aire(self) -> float:
        """
        Aire du cercle.

        Retour
        ------
        float
        """
        raise NotImplementedError("A COMPLETER")

    def perimetre(self) -> float:
        """
        Périmètre du cercle.

        Retour
        ------
        float
        """
        raise NotImplementedError("A COMPLETER")

    def boite_englobante(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante du cercle : (x−r, y−r, x+r, y+r).

        Retour
        ------
        tuple[float,float,float,float]
        """
        raise NotImplementedError("A COMPLETER")

    def contient(self, p: Point) -> bool:
        """
        Test d'appartenance (bords inclus).

        Paramètres
        ----------
        p : Point

        Retour
        ------
        bool
        """
        raise NotImplementedError("A COMPLETER")

    def transformer(self, T: Transformation2D) -> "Cercle":
        """
        Retourne un *nouveau* cercle transformé.

        Règle (simplifiée)
        ------------------
        Le centre devient T(centre).
        Le rayon est mis à l'échelle de façon isotrope avec sqrt(|det(M)|),
        où M est la matrice 2x2 de T.

        Paramètres
        ----------
        T : Transformation2D

        Retour
        ------
        Cercle
        """
        raise NotImplementedError("A COMPLETER")

    def tracer(self, ax) -> None:
        """
        Trace le cercle (sans remplissage) sur l'Axes donné.

        Paramètres
        ----------
        ax : matplotlib.axes.Axes
        """
        c = patches.Circle((self._centre.obtenir_x(), self._centre.obtenir_y()),
                           self._rayon, fill=False)
        ax.add_patch(c)


# ---------------------------------------------------------------------
# 6) Rectangle
# ---------------------------------------------------------------------
class Rectangle(Figure):
    """
    Rectangle *aligné sur les axes*, défini par un coin inférieur gauche,
    une largeur (>0) et une hauteur (>0).

    Attributs privés
    ----------------
    _coin    : Point
    _largeur : float
    _hauteur : float
    """

    def __init__(self, coin: Point, largeur: float, hauteur: float):
        """
        Construit un rectangle.

        Paramètres
        ----------
        coin : Point
            Coin inférieur gauche (par convention).
        largeur, hauteur : float
            Strictement positives.

        Exceptions
        ----------
        ValueError : si largeur <= 0 ou hauteur <= 0.
        """
        raise NotImplementedError("A COMPLETER")

    def obtenir_coin(self) -> Point:
        """Retourne le coin inférieur gauche."""
        raise NotImplementedError("A COMPLETER")

    def obtenir_largeur(self) -> float:
        """Retourne la largeur du rectangle."""
        raise NotImplementedError("A COMPLETER")

    def obtenir_hauteur(self) -> float:
        """Retourne la hauteur du rectangle."""
        raise NotImplementedError("A COMPLETER")

    def definir_largeur(self, L: float) -> None:
        """
        Modifie la largeur.

        Paramètres
        ----------
        L : float (>0)

        Exceptions
        ----------
        ValueError : si L <= 0.
        """
        raise NotImplementedError("A COMPLETER")

    def definir_hauteur(self, H: float) -> None:
        """
        Modifie la hauteur.

        Paramètres
        ----------
        H : float (>0)

        Exceptions
        ----------
        ValueError : si H <= 0.
        """
        raise NotImplementedError("A COMPLETER")

    # --- Figure ---

    def aire(self) -> float:
        """Aire du rectangle."""
        raise NotImplementedError("A COMPLETER")

    def perimetre(self) -> float:
        """Périmètre du rectangle."""
        raise NotImplementedError("A COMPLETER")

    def boite_englobante(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante.

        Retour
        ------
        tuple[float,float,float,float]
        """
        raise NotImplementedError("A COMPLETER")

    def contient(self, p: Point) -> bool:
        """
        Test d'appartenance (bords inclus, tolérance EPS).

        Critère
        -------
        xmin−EPS <= px <= xmax+EPS
        ymin−EPS <= py <= ymax+EPS

        Paramètres
        ----------
        p : Point

        Retour
        ------
        bool
        """
        raise NotImplementedError("A COMPLETER")

    def translater(self, dx: float, dy: float) -> None:
        """
        Translate le rectangle en déplaçant son coin (modifie *en place*).

        Paramètres
        ----------
        dx, dy : float
        """
        raise NotImplementedError("A COMPLETER")

    def mettre_echelle(self, kx: float, ky: float) -> None:
        """
        Mise à l'échelle anisotrope du rectangle (modifie *en place*).

        Paramètres
        ----------
        kx, ky : float
            Facteurs d'échelle (> 0).

        Exceptions
        ----------
        ValueError : si kx <= 0 ou ky <= 0.
        """
        raise NotImplementedError("A COMPLETER")

    def transformer(self, T: Transformation2D) -> "Rectangle":
        """
        Retourne un *nouveau* rectangle représentant la boite englobante
        des 4 sommets transformés par T.

        Détails
        -------
        - On applique T aux 4 sommets.
        - On renvoie la boîte englobante de ces points.

        Paramètres
        ----------
        T : Transformation2D

        Retour
        ------
        Rectangle
        """
        raise NotImplementedError("A COMPLETER")

    def tracer(self, ax) -> None:
        """
        Trace le rectangle (sans remplissage) sur l'Axes donné.

        Paramètres
        ----------
        ax : matplotlib.axes.Axes
        """
        r = patches.Rectangle((self._coin.obtenir_x(), self._coin.obtenir_y()),
                              self._largeur, self._hauteur, fill=False)
        ax.add_patch(r)


# ---------------------------------------------------------------------
# 7) Scene
# ---------------------------------------------------------------------
class Scene:
    """
    Agrège des figures, fournit des métriques globales et des fonctions de tracé.

    Attributs privés
    ----------------
    _figures : list[Figure]
        Collection de figures (copiée à la lecture pour éviter les effets de bord).
    """

    def __init__(self):
        """Construit une scène vide."""
        raise NotImplementedError("A COMPLETER")

    def ajouter(self, f: Figure) -> None:
        """
        Ajoute une figure à la scène.

        Paramètres
        ----------
        f : Figure
        """
        raise NotImplementedError("A COMPLETER")

    def obtenir_figures(self) -> List[Figure]:
        """
        Retourne une *copie* de la liste des figures.

        Retour
        ------
        list[Figure]
        """
        raise NotImplementedError("A COMPLETER")

    def aire_totale(self) -> float:
        """
        Somme des aires de toutes les figures de la scène.

        Retour
        ------
        float
        """
        raise NotImplementedError("A COMPLETER")

    def perimetre_total(self) -> float:
        """
        Somme des périmètres des figures.

        Retour
        ------
        float
        """
        raise NotImplementedError("A COMPLETER")

    def boite_scene(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante globale de la scène.

        Retour
        ------
        (xmin, ymin, xmax, ymax) : tuple[float,float,float,float]
            Si la scène est vide : (0,0,0,0).
        """
        raise NotImplementedError("A COMPLETER")

    def translater_tout(self, dx: float, dy: float) -> None:
        """
        Translate *toutes* les figures en appliquant la translation (dx,dy)
        via leur méthode `transformer(T)`.

        Paramètres
        ----------
        dx, dy : float
        """
        raise NotImplementedError("A COMPLETER")

    def tracer(self, montrer_boite: bool = True) -> None:
        """
        Trace toutes les figures de la scène. Optionnellement, affiche la
        boîte englobante globale en pointillé.

        Paramètres
        ----------
        montrer_boite : bool, par défaut True
            Affiche la bbox de scène si True.
        """
        fig, ax = plt.subplots()
        for f in self._figures:
            f.tracer(ax)
        if montrer_boite and self._figures:
            x0, y0, x1, y1 = self.boite_scene()
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k:", lw=1)
        if self._figures:
            x0, y0, x1, y1 = self.boite_scene()
            pad = 0.2 * max(1.0, (x1 - x0) + (y1 - y0))
            ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, linestyle=":")
        ax.set_title("Scene : figures")
        plt.show()
