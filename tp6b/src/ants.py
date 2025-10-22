#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple
import numpy as np
import random

class CellType(Enum):
    EMPTY = auto()
    NEST  = auto()
    FOOD  = auto()
    WALL  = auto()

@dataclass
class Params:
    width: int = 100
    height: int = 70
    n_ants: int = 200
    # phéromones
    evap_rate: float = 0.02
    deposit_amount: float = 0.01
    deposit_amount_return: float = 2.0
    diffuse_rate: float = 0.05
    # déplacement
    alpha: float = 2.0
    explore_eps: float = 0.04
    neighborhood: str = "8"    # "4" ou "8"
    home_bias: float = 1.5
    # aléa
    seed: int = 1

class World:
    def __init__(self, params: Params, nest: Tuple[int,int], foods: List[Tuple[int,int]]) -> None:
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.W, self.H = self.p.width, self.p.height
        self.grid = np.full((self.H, self.W), CellType.EMPTY, dtype=object)
        self.nest = nest
        self.foods = foods[:]
        self.grid[nest[1], nest[0]] = CellType.NEST
        for fx, fy in self.foods:
            self.grid[fy, fx] = CellType.FOOD
        self.pher = np.zeros((self.H, self.W), dtype=np.float32)
        self.blocked = np.zeros((self.H, self.W), dtype=bool)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def is_free(self, x: int, y: int) -> bool:
        # TODO[1] : retour True si la case est dans la grille ET non bloquée
        ...

    def add_rect_obstacle(self, x0: int, y0: int, w: int, h: int) -> None:
        # TODO[2] : marquer blocked[y0:y0+h, x0:x0+w] = True
        #           puis s'assurer que "nest" et "foods" restent libres
        ...

    def step_pheromone(self, deposits: List[Tuple[int,int,float]]) -> None:
        # TODO[3] : appliquer les dépôts (x,y,q) sur self.pher
        # TODO[4] : évaporation self.pher *= (1.0 - self.p.evap_rate)
        # TODO[5] : diffusion 8-connexes si self.p.diffuse_rate > 0
        ...

class Ant:
    __slots__ = ("x","y","has_food","path_stack","rng")

    def __init__(self, x: int, y: int, rng: random.Random) -> None:
        self.x = x
        self.y = y
        self.has_food = False
        self.path_stack: List[Tuple[int,int]] = []
        self.rng = rng

    def neighbors(self, world: World) -> List[Tuple[int,int]]:
        # TODO[6] : retourner les voisins libres (4 ou 8 selon world.p.neighborhood)
        ...

    def step(self, world: World) -> Tuple[int,int,float]:
        # TODO[7] : événements FOOD/NEST (changement de has_food, reset de path_stack)
        # TODO[8] : si retour et path_stack non vide -> backtracking + dépôt retour
        # TODO[9] : sinon, choix epsilon-greedy (exploration) ou pondéré par (eps+pher)^alpha
        #           si has_food, multiplier les voisins qui rapprochent du nid par home_bias
        # TODO[10]: mise à jour position, mémoire du chemin (à l'aller), quantité déposée
        ...
        # retourner (x, y, q_depose)

class Simulation:
    def __init__(self, params: Params, nest: Tuple[int,int], foods: List[Tuple[int,int]]) -> None:
        self.p = params
        random.seed(self.p.seed)
        self.world = World(self.p, nest, foods)
        self.ants = [Ant(nest[0], nest[1], random.Random(self.p.seed + i))
                     for i in range(self.p.n_ants)]
        self.t = 0

    def step(self) -> None:
        deposits: List[Tuple[int,int,float]] = []
        for ant in self.ants:
            x, y, q = ant.step(self.world)
            if q > 0.0:
                deposits.append((x, y, q))
        self.world.step_pheromone(deposits)
        self.t += 1
