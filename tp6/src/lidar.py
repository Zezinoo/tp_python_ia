#!/usr/bin/env python3

"""
====================
     LiDAR 2D
====================

Ce module fournit :
- MesureLidar : une mesure angulaire (theta) et une distance (ou None)
- Lidar2D     : un capteur 2D qui balaie un champ angulaire et interroge la scène

Dépendances attendues :
- src/geometrie.py : Point, Vecteur2D, Scene (TP POO n°1)
- src/optique.py   : figures optiques (CercleOptique, RectangleOptique, ...),
                     et surtout la présence d'une méthode `intersection_rayon(o, d)`
                     sur les figures, qui renvoie (t, P, n) ou None.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import cos, sin
from typing import List, Optional

# Adaptation au fait que les modules sont dans "src/"
import os, sys, random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import des briques géométriques
from src.geometrie import Point, Vecteur2D, Scene


@dataclass
class MesureLidar:
    """
    Une mesure LiDAR 2D : angle (rad) + distance (m) ou None si pas d'impact.

    Attributs
    ---------
    angle : float
        Angle relatif à l'orientation du capteur (en radians).
    distance : float | None
        Distance mesurée. None s'il n'y a pas d'intersection dans la scène.

    Remarque
    --------
    L'angle est *relatif* à l'orientation du capteur. L'angle absolu utilisé
    pour lancer le rayon vaut : theta_abs = orientation + angle.
    """
    pass

class Lidar2D:
    """
    Capteur LiDAR 2D minimaliste avec bruit de mesure optionnel.

    Attributs
    ---------
    position : Point
        Position du capteur dans la scène.
    orientation : float
        Orientation de référence (en radians). Les angles de balayage s'y ajoutent.
    champ_de_vision : float
        Champ de vision en radians (p. ex. 2*PI pour un tour complet).
    resolution : int
        Nombre de rayons (échantillons angulaires) dans le champ de vision.
    distance_max : float
        Distance maximale de mesure (au-delà : pas de retour = None).
    sigma_bruit : float
        Écart-type du bruit gaussien ajouté aux distances valides (m). 0.0 = pas de bruit.

    Méthodes principales
    --------------------
    - balayer(scene) -> List[MesureLidar]
        Lance `resolution` rayons répartis uniformément sur `champ_de_vision`, retourne
        la liste (angle relatif, distance bruitée ou None).
    - mesures_en_points(mesures) -> List[Point]
        Convertit les mesures polaires (angle relatif, distance) en points (x, y)
        dans le repère de la scène.
    """
    pass
