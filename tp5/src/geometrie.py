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

# nécessaire pour que l'interpréteur ne génère pas d'erreur lorsqu'une classe 
# est déclarée sans avoir été encore définie
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt, pi, cos, sin
from typing import Tuple, List

import numpy as np
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

        new_x = v.x + self.x
        new_y = v.y + self.y

        return Vecteur2D(x = new_x , y = new_y)
        ...

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

        new_x = self.x - v.x
        new_y = self.y - v.y

        return Vecteur2D(x = new_x , y = new_y)
        ...

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

        new_x = k * self.x
        new_y = k * self.y

        return Vecteur2D(x = new_x , y = new_y)
        ...

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

        new_x = v.x * self.x
        new_y = v.y * self.y

        return new_x + new_y
        ...

    def norme(self) -> float:
        """
        Calcule la norme euclidienne du vecteur.

        Retour
        ------
        float
        """
        square_sum = self.x **2 + self.y ** 2
        norm = np.sqrt(square_sum)

        return norm
        ...

    def normaliser(self) -> "Vecteur2D":
        """
        Retourne un vecteur unitaire de même direction.

        Retour
        ------
        Vecteur2D
            Vecteur normalisé. Si la norme vaut 0, retourner (0,0).
        """
        norm = self.norme()
        if norm == 0:
            new_x = 0
            new_y = 0
            print("Zero normed vector! (0,0) returned")
            return Vecteur2D(0 , 0)
        
        new_x = self.x/norm
        new_y = self.y / norm
        return Vecteur2D(x = new_x , y = new_y)

        ...

    def perpendiculaire_gauche(self , theta = np.pi/2) -> "Vecteur2D":
        """
        Retourne le vecteur perpendiculaire gauche (rotation +90°).

        Retour
        ------
        Vecteur2D
        """

        return Vecteur2D(-self.y , self.x)
        ...

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
        self.__x = x
        self.__y = y
        ...

    def obtenir_x(self) -> float:
        """
        Lit l'abscisse du point.

        Retour
        ------
        float
            Valeur de x.
        """
        return self.__x
        ...

    def obtenir_y(self) -> float:
        """
        Lit l'ordonnée du point.

        Retour
        ------
        float
            Valeur de y.
        """
        return self.__y
        ...

    def definir_x(self, v: float) -> None:
        """
        Modifie l'abscisse du point.

        Paramètres
        ----------
        v : float
            Nouvelle valeur de x.
        """
        self.__x = v
        ...

    def definir_y(self, v: float) -> None:
        """
        Modifie l'ordonnée du point.

        Paramètres
        ----------
        v : float
            Nouvelle valeur de y.
        """
        self.__y = v
        ...

    def translater(self, dx: float, dy: float) -> None:
        """
        Translate le point par (dx, dy) (modifie *en place*).

        Paramètres
        ----------
        dx, dy : float
            Incréments à ajouter à (x, y).
        """
        self.__x += dx
        self.__y += dy
        ...

    def vers_vecteur(self) -> Vecteur2D:
        """
        Convertit le point en vecteur (x, y).

        Retour
        ------
        Vecteur2D
        """
        return Vecteur2D(x = self.__x , y = self.__y)
        ...

    def __repr__(self) -> str:
        """
        Représentation lisible (utile pour le débogage).

        Retour
        ------
        str
            Exemple : "Point(x=1.000, y=2.000)".
        """
        return f"Point(x={self.__x:.2f} , y={self.__y:.2f})"
        ...


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
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.tx = tx
        self.ty = ty
        ...

    @staticmethod
    def identite() -> "Transformation2D":
        """
        Fabrique la transformation identité.

        Retour
        ------
        Transformation2D
        """
        return Transformation2D(1 , 0 , 0, 1 , 0 ,0)
        ...

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
        return Transformation2D(1,0,0,1,dx , dy)
        ...

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
        a = np.cos(theta_rad)
        b = -np.sin(theta_rad)
        c = np.sin(theta_rad)
        d = np.cos(theta_rad)

        return Transformation2D(a , b, c, d , 0 ,0)
        ...

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
        return Transformation2D(sx , 0 , 0 , sy , 0 , 0)
        ...

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
        new_x = self.a * p.obtenir_x()  + self.b * p.obtenir_y() + self.tx
        new_y = self.c * p.obtenir_x() + self.d * p.obtenir_y() + self.ty
        return Point(x = new_x , y = new_y)
        ...

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
        new_a = self.a * autre.a + self.b * autre.c
        new_b = self.a * autre.b + self.b * autre.d
        new_c = self.c * autre.a + self.d * autre.c
        new_d = self.c * autre.b + self.d * autre.d

        new_tx = self.tx + self.a * autre.tx + self.b * autre.ty
        new_ty = self.ty + self.c * autre.tx + self.d * autre.ty

        return Transformation2D(new_a , new_b , new_c , new_d , new_tx , new_ty)


        ...


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
        ...

    @abstractmethod
    def perimetre(self) -> float:
        """
        Périmètre (longueur du contour).

        Retour
        ------
        float
            Périmètre (>= 0).
        """
        ...

    @abstractmethod
    def boite_englobante(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante axis-aligned de la figure.

        Retour
        ------
        (xmin, ymin, xmax, ymax) : tuple[float,float,float,float]
            Avec xmin ≤ xmax, ymin ≤ ymax.
        """
        ...

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
        ...

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
        ...

    @abstractmethod
    def tracer(self, ax) -> None:
        """
        Dessine la figure dans un Axes matplotlib.

        Paramètres
        ----------
        ax : matplotlib.axes.Axes
            Axes cible. Utiliser patches.Circle / patches.Rectangle, fill=False.
        """
        ...


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
        self.__centre = centre
        self.__rayon = rayon
        
        if rayon <= 0:
            raise ValueError("Radius must be greater than zero!")
        ...

    def obtenir_centre(self) -> Point:
        """
        Retourne le centre du cercle.

        Retour
        ------
        Point
        """
        return self.__centre
        ...

    def obtenir_rayon(self) -> float:
        """
        Retourne le rayon du cercle.

        Retour
        ------
        float
        """
        return self.__rayon
        ...

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
        if r <= 0 :
            raise ValueError("Radius must be greater than zero!")
        self.__rayon = r
        ...

    # --- Figure ---

    def aire(self) -> float:
        """
        Aire du cercle.

        Retour
        ------
        float
        """

        return np.pi * self.__rayon **2 
        ...

    def perimetre(self) -> float:
        """
        Périmètre du cercle.

        Retour
        ------
        float
        """
        return 2 * np.pi * self.__rayon
        ...

    def boite_englobante(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante du cercle : (x−r, y−r, x+r, y+r).

        Retour
        ------
        tuple[float,float,float,float]
        """
        x = self.__centre.obtenir_x()
        y = self.__centre.obtenir_y()
        r = self.obtenir_rayon()

        return(x-r , y - r , x+r , y + r)
        ...

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
        isin_circle = None

        x_center = self.obtenir_centre().obtenir_x()
        y_center = self.obtenir_centre().obtenir_y()
        vector_center_to_point = Vecteur2D(x = p.obtenir_x() - x_center , y = p.obtenir_y() - y_center)
        norm = vector_center_to_point.norme()

        if abs(norm) > self.obtenir_rayon():
            isin_circle = False
        else:
            isin_circle = True

        return isin_circle
        ...

    def transformer(self, T: Transformation2D) -> "Cercle":
        """
        Retourne un *nouveau* cercle transformé.

        Règle (simplifiée)
        ------------------
        Le centre devient T(centre).
        Le rayon est mis à l'échelle de façon *isotrope* avec sqrt(|det(M)|),
        où M est la matrice 2x2 de T. (On ne gère pas l'ellipse ici.)

        Paramètres
        ----------
        T : Transformation2D

        Retour
        ------
        Cercle
        """

        abs_det = abs(T.a * T.d - T.b * T.c)
        sqrt_det = np.sqrt(abs_det)

        center = self.obtenir_centre()
        new_center = T.appliquer_point(center)
        new_radius = sqrt_det * self.obtenir_rayon()

        return Cercle(centre = new_center , rayon=new_radius)
        ...

    def tracer(self, ax) -> None:
        """
        Trace le cercle (sans remplissage) sur l'Axes donné.

        Paramètres
        ----------
        ax : matplotlib.axes.Axes
        """
        c = patches.Circle((self.__centre.obtenir_x(), self.__centre.obtenir_y()),
                           self.__rayon, fill=False)
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

        self.__coin = coin
        self.__largeur = largeur
        self.__hauteur = hauteur
        
        if largeur <= 0:
            raise ValueError("Width must be greater than 0")
        if hauteur <= 0:
            raise ValueError("Width must be greater than 0")

        ...

    def obtenir_coin(self) -> Point:
        """Retourne le coin inférieur gauche."""
        return self.__coin
        ...

    def obtenir_largeur(self) -> float:
        """Retourne la largeur du rectangle."""
        return self.__largeur
        ...

    def obtenir_hauteur(self) -> float:
        """Retourne la hauteur du rectangle."""
        return self.__hauteur
        ...

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
        if L <= 0:
            raise ValueError("Width must be > 0.")
        self.__largeur = L
        ...

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
        if H <= 0:
            raise ValueError("Height must be > 0.")

        self.__hauteur = H 
        ...

    # --- Figure ---

    def aire(self) -> float:
        """Aire du rectangle."""
        return self.__hauteur * self.__largeur
        ...

    def perimetre(self) -> float:
        """Périmètre du rectangle."""
        return 2*self.__largeur + 2*self.__hauteur
        ...

    def boite_englobante(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante.

        Retour
        ------
        tuple[float,float,float,float]
        """
        x_c = self.obtenir_coin().obtenir_x()
        y_c = self.obtenir_coin().obtenir_y()


        return (x_c , y_c , x_c + self.obtenir_largeur() , y_c + self.obtenir_hauteur() )
        ...

    def contient(self, p: Point , eps = 1e-3) -> bool:
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

        isin_rect = None

        x_c = self.obtenir_coin().obtenir_x()
        y_c = self.obtenir_coin().obtenir_y()

        xmin = x_c
        xmax = x_c + self.obtenir_largeur()

        ymin = y_c
        ymax = y_c + self.obtenir_hauteur()

        p_x = p.obtenir_x()
        p_y = p.obtenir_y()

        isin_x = (p_x >= xmin - eps) & (p_x <= xmax + eps)
        isin_y = (p_y >= ymin - eps) & (p_y <= ymax + eps)

        if isin_x and isin_y:
            isin_rect = True
        else:
            isin_rect = False

        return isin_rect
        
        
        ...

    def translater(self, dx: float, dy: float) -> None:
        """
        Translate le rectangle en déplaçant son coin (modifie *en place*).

        Paramètres
        ----------
        dx, dy : float
        """
        x_c = self.obtenir_coin().obtenir_x()
        y_c = self.obtenir_coin().obtenir_y()

        x_new = x_c + dx
        y_new = y_c + dy

        self.__coin = Point(x_new , y_new)
        ...

    def mettre_echelle(self, kx: float, ky: float) -> None:
        """
        Mise à l'échelle anisotrope du rectangle (modifie *en place*).

        Paramètres
        ----------
        kx, ky : float
            Facteurs d'échelle (>0).

        Exceptions
        ----------
        ValueError : si kx <= 0 ou ky <= 0.
        """
        if kx <= 0:
            raise ValueError("Kx must be greater than zero!")
        if ky <= 0:
            raise ValueError("Ky must be greater than zero!")
        
        self.definir_hauteur(self.obtenir_hauteur() * ky)
        self.definir_largeur(self.obtenir_largeur() * kx)
        ...

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
        x, y = self.obtenir_coin().obtenir_x(), self.obtenir_coin().obtenir_y()
        w, h = self.obtenir_largeur(), self.obtenir_hauteur()
        corners = [
            Point(x,     y),
            Point(x+w,   y),
            Point(x,   y+h),
            Point(x+w, y+h),
        ]
        tc = [T.appliquer_point(p) for p in corners]
        xs = [p.obtenir_x() for p in tc]
        ys = [p.obtenir_y() for p in tc]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return Rectangle(Point(xmin, ymin), xmax - xmin, ymax - ymin)
        ...

    def tracer(self, ax) -> None:
        """
        Trace le rectangle (sans remplissage) sur l'Axes donné.

        Paramètres
        ----------
        ax : matplotlib.axes.Axes
        """
        print(self.__coin.obtenir_x())
        print(self.__coin.obtenir_y())
        r = patches.Rectangle((self.__coin.obtenir_x(), self.__coin.obtenir_y()),
                              self.__largeur, self.__hauteur, fill=False)
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
        ...
        self.__figures = []

    def ajouter(self, f: Figure) -> None:
        """
        Ajoute une figure à la scène.

        Paramètres
        ----------
        f : Figure
        """
        self.__figures.append(f)
        ...

    def obtenir_figures(self) -> List[Figure]:
        """
        Retourne une *copie* de la liste des figures.

        Retour
        ------
        list[Figure]
        """
        return self.__figures.copy()
        ...

    def aire_totale(self) -> float:
        """
        Somme des aires de toutes les figures de la scène.

        Retour
        ------
        float
        """
        total = 0
        for f in self.__figures:
            total += f.aire()
        
        return total

        ...

    def perimetre_total(self) -> float:
        """
        Somme des périmètres des figures.

        Retour
        ------
        float
        """

        total = 0
        for f in self.__figures:
            total += f.perimetre()
        
        return total
        ...

    def boite_scene(self) -> Tuple[float, float, float, float]:
        """
        Boîte englobante globale de la scène.

        Retour
        ------
        (xmin, ymin, xmax, ymax) : tuple[float,float,float,float]
            Si la scène est vide : (0,0,0,0).
        """
        if not self.__figures:
            return (0.0, 0.0, 0.0, 0.0)
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        for f in self.__figures:
            x0, y0, x1, y1 = f.boite_englobante()
            xmin = min(xmin, x0); ymin = min(ymin, y0)
            xmax = max(xmax, x1); ymax = max(ymax, y1)
        return (xmin, ymin, xmax, ymax)

        ...

    def translater_tout(self, dx: float, dy: float) -> None:
        """
        Translate *toutes* les figures en appliquant la translation (dx,dy)
        via leur méthode `transformer(T)`.

        Paramètres
        ----------
        dx, dy : float
        """
        new_figures = []
        T = Transformation2D(1,0,0,1,dx,dy)
        for f in self.__figures:
            new_f = f.transformer(T)
            new_figures.append(new_f)
        
        self.__figures = new_figures
        ...

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
        for f in self.__figures:
            f.tracer(ax)
        if montrer_boite and self.__figures:
            x0, y0, x1, y1 = self.boite_scene()
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k:", lw=1)
        if self.__figures:
            x0, y0, x1, y1 = self.boite_scene()
            pad = 0.2 * max(1.0, (x1 - x0) + (y1 - y0))
            ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, linestyle=":")
        ax.set_title("Scene : figures")
        plt.show()
