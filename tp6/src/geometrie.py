# src/geometrie.py

from __future__ import annotations
from abc import ABC, abstractmethod
from math import sqrt, pi, cos, sin
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# -----------------------------
# Outils geometriques 2D
# -----------------------------
@dataclass
class Vecteur2D:
    x: float
    y: float

    def ajouter(self, v: "Vecteur2D") -> "Vecteur2D":
        return Vecteur2D(self.x + v.x, self.y + v.y)

    def soustraire(self, v: "Vecteur2D") -> "Vecteur2D":
        return Vecteur2D(self.x - v.x, self.y - v.y)

    def multiplier(self, k: float) -> "Vecteur2D":
        return Vecteur2D(self.x * k, self.y * k)

    def produit_scalaire(self, v: "Vecteur2D") -> float:
        return self.x * v.x + self.y * v.y

    def norme(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def normaliser(self) -> "Vecteur2D":
        n = self.norme()
        return Vecteur2D(self.x / n, self.y / n) if n > 0 else Vecteur2D(0.0, 0.0)

    def perpendiculaire_gauche(self) -> "Vecteur2D":
        return Vecteur2D(-self.y, self.x)


class Point:
    def __init__(self, x: float, y: float):
        self._x = float(x)
        self._y = float(y)

    def obtenir_x(self) -> float: return self._x
    def obtenir_y(self) -> float: return self._y
    def definir_x(self, v: float) -> None: self._x = float(v)
    def definir_y(self, v: float) -> None: self._y = float(v)

    def vers_vecteur(self) -> Vecteur2D: return Vecteur2D(self._x, self._y)

    def translater(self, dx: float, dy: float) -> None:
        self._x += dx; self._y += dy

    def __repr__(self) -> str:
        return f"Point(x={self._x:.3f}, y={self._y:.3f})"


# -----------------------------
# Transformations affines 2D
# -----------------------------
class Transformation2D:
    """
    Représente une transformation affine 2D sous forme de matrice :
       | a  b  tx |
       | c  d  ty |
       | 0  0   1 |
    P' = M * P + t
    """
    def __init__(self, a: float, b: float, c: float, d: float, tx: float, ty: float):
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)
        self.tx, self.ty = float(tx), float(ty)

    @staticmethod
    def identite() -> "Transformation2D":
        return Transformation2D(1, 0, 0, 1, 0, 0)

    @staticmethod
    def translation(dx: float, dy: float) -> "Transformation2D":
        return Transformation2D(1, 0, 0, 1, dx, dy)

    @staticmethod
    def rotation(theta_rad: float) -> "Transformation2D":
        c, s = cos(theta_rad), sin(theta_rad)
        return Transformation2D(c, -s, s, c, 0, 0)

    @staticmethod
    def mise_echelle(sx: float, sy: float) -> "Transformation2D":
        return Transformation2D(sx, 0, 0, sy, 0, 0)

    def appliquer_point(self, p: Point) -> Point:
        x, y = p.obtenir_x(), p.obtenir_y()
        return Point(self.a * x + self.b * y + self.tx,
                     self.c * x + self.d * y + self.ty)

    def composer(self, autre: "Transformation2D") -> "Transformation2D":
        # self ∘ autre  (applique d'abord 'autre', puis 'self')
        a = self.a * autre.a + self.b * autre.c
        b = self.a * autre.b + self.b * autre.d
        c = self.c * autre.a + self.d * autre.c
        d = self.c * autre.b + self.d * autre.d
        tx = self.a * autre.tx + self.b * autre.ty + self.tx
        ty = self.c * autre.tx + self.d * autre.ty + self.ty
        return Transformation2D(a, b, c, d, tx, ty)


# -----------------------------
# POO : Figures (sans optique)
# -----------------------------
class Figure(ABC):
    @abstractmethod
    def aire(self) -> float: ...
    @abstractmethod
    def perimetre(self) -> float: ...
    @abstractmethod
    def boite_englobante(self) -> tuple[float, float, float, float]: ...
    @abstractmethod
    def contient(self, p: Point) -> bool: ...
    @abstractmethod
    def transformer(self, T: Transformation2D) -> "Figure": ...
    @abstractmethod
    def tracer(self, ax) -> None: ...


class Cercle(Figure):
    def __init__(self, centre: Point, rayon: float):
        if rayon <= 0:
            raise ValueError("rayon doit etre > 0")
        self._centre = centre
        self._rayon = float(rayon)

    def obtenir_centre(self) -> Point: return self._centre
    def obtenir_rayon(self) -> float: return self._rayon
    def definir_rayon(self, r: float) -> None:
        if r <= 0: raise ValueError("rayon > 0 requis")
        self._rayon = float(r)

    def aire(self) -> float: return pi * self._rayon * self._rayon
    def perimetre(self) -> float: return 2 * pi * self._rayon

    def boite_englobante(self) -> tuple[float, float, float, float]:
        x, y, r = self._centre.obtenir_x(), self._centre.obtenir_y(), self._rayon
        return (x - r, y - r, x + r, y + r)

    def contient(self, p: Point) -> bool:
        v = p.vers_vecteur().soustraire(self._centre.vers_vecteur())
        return v.produit_scalaire(v) <= self._rayon * self._rayon + 1e-12

    def mettre_echelle(self, k: float) -> None:
        if k <= 0: raise ValueError("facteur d'echelle > 0")
        self._rayon *= k

    def transformer(self, T: Transformation2D) -> "Cercle":
        c2 = T.appliquer_point(self._centre)
        # approx isotrope (sqrt(|det(M)|)) — pas d'ellipse dans ce TP
        approx_scale = sqrt(abs(T.a * T.d - T.b * T.c))
        r2 = self._rayon * (approx_scale if approx_scale > 0 else 1.0)
        return Cercle(c2, r2)

    def tracer(self, ax) -> None:
        c = patches.Circle((self._centre.obtenir_x(), self._centre.obtenir_y()),
                           self._rayon, fill=False)
        ax.add_patch(c)

    def __repr__(self) -> str:
        return f"Cercle(centre={self._centre}, rayon={self._rayon:.3f})"


class Rectangle(Figure):
    """
    Rectangle aligne sur les axes (coin bas-gauche, largeur, hauteur).
    """
    def __init__(self, coin: Point, largeur: float, hauteur: float):
        if largeur <= 0 or hauteur <= 0:
            raise ValueError("largeur/hauteur doivent etre > 0")
        self._coin = coin
        self._largeur = float(largeur)
        self._hauteur = float(hauteur)

    def obtenir_coin(self) -> Point: return self._coin
    def obtenir_largeur(self) -> float: return self._largeur
    def obtenir_hauteur(self) -> float: return self._hauteur
    def definir_largeur(self, L: float) -> None:
        if L <= 0: raise ValueError("largeur > 0")
        self._largeur = float(L)
    def definir_hauteur(self, H: float) -> None:
        if H <= 0: raise ValueError("hauteur > 0")
        self._hauteur = float(H)

    def aire(self) -> float: return self._largeur * self._hauteur
    def perimetre(self) -> float: return 2 * (self._largeur + self._hauteur)

    def boite_englobante(self) -> tuple[float, float, float, float]:
        x, y = self._coin.obtenir_x(), self._coin.obtenir_y()
        return (x, y, x + self._largeur, y + self._hauteur)

    def contient(self, p: Point) -> bool:
        x, y = p.obtenir_x(), p.obtenir_y()
        xmin, ymin, xmax, ymax = self.boite_englobante()
        eps = 1e-12
        return (xmin - eps) <= x <= (xmax + eps) and (ymin - eps) <= y <= (ymax + eps)

    def translater(self, dx: float, dy: float) -> None:
        self._coin.translater(dx, dy)

    def mettre_echelle(self, kx: float, ky: float) -> None:
        if kx <= 0 or ky <= 0: raise ValueError("facteurs d'echelle > 0")
        self._largeur *= kx; self._hauteur *= ky

    def transformer(self, T: Transformation2D) -> "Rectangle":
        # Applique T aux 4 sommets et retourne la bbox axis-aligned
        x, y = self._coin.obtenir_x(), self._coin.obtenir_y()
        pts = [Point(x, y),
               Point(x + self._largeur, y),
               Point(x, y + self._hauteur),
               Point(x + self._largeur, y + self._hauteur)]
        tpts = [T.appliquer_point(p) for p in pts]
        xs = [p.obtenir_x() for p in tpts]; ys = [p.obtenir_y() for p in tpts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return Rectangle(Point(xmin, ymin), xmax - xmin, ymax - ymin)

    def tracer(self, ax) -> None:
        r = patches.Rectangle((self._coin.obtenir_x(), self._coin.obtenir_y()),
                              self._largeur, self._hauteur, fill=False)
        ax.add_patch(r)

    def __repr__(self) -> str:
        return f"Rectangle(coin={self._coin}, L={self._largeur:.3f}, H={self._hauteur:.3f})"


# -----------------------------
# Scene (sans optique)
# -----------------------------
class Scene:
    def __init__(self):
        self._figures: list[Figure] = []

    def ajouter(self, f: Figure) -> None:
        self._figures.append(f)

    def obtenir_figures(self) -> list[Figure]:
        return list(self._figures)

    def aire_totale(self) -> float:
        return sum(f.aire() for f in self._figures)

    def perimetre_total(self) -> float:
        return sum(f.perimetre() for f in self._figures)

    def boite_scene(self) -> tuple[float, float, float, float]:
        if not self._figures:
            return (0, 0, 0, 0)
        xs, ys, Xs, Ys = [], [], [], []
        for f in self._figures:
            x0, y0, x1, y1 = f.boite_englobante()
            xs.append(x0); ys.append(y0); Xs.append(x1); Ys.append(y1)
        return (min(xs), min(ys), max(Xs), max(Ys))

    def translater_tout(self, dx: float, dy: float) -> None:
        T = Transformation2D.translation(dx, dy)
        self._figures = [f.transformer(T) for f in self._figures]

    def tracer(self, montrer_boite: bool = True) -> None:
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
