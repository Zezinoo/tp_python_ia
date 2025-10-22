#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple
import numpy as np
import random
import scipy

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
        # x cooord is columns , y coord is columns
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
        self.blocked = np.zeros((self.H, self.W), dtype=bool) # is initalized as false, meaning FALSE is not blocked

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def is_free(self, x: int, y: int) -> bool:
        is_in_bounds = self.in_bounds(x ,  y)
        return is_in_bounds and not(self.blocked[y,x])
        
        # TODO[1] : retour True si la case(cell) est dans la grille ET non bloquée
        ...

    def add_rect_obstacle(self, x0: int, y0: int, w: int, h: int) -> None:
        # TODO[2] : marquer blocked[y0:y0+h, x0:x0+w] = True
        #           puis s'assurer que "nest" et "foods" restent libres
        ...
        self.blocked[y0 : y0 + h , x0 : x0 + w] = True
        nx, ny = self.nest
        self.blocked[ny, nx] = False
        for fx , fy in self.foods:
            self.blocked[fy , fx] = False # Vectorize this loop using masks



    def step_pheromone(self, deposits: List[Tuple[int,int,float]]) -> None:
        # TODO[3] : appliquer les dépôts (x,y,q) sur self.pher
        for px , py , q in deposits:
            self.pher[py , px ] += q
        # TODO[4] : évaporation self.pher *= (1.0 - self.p.evap_rate)
        self.pher *=  1 - self.p.evap_rate
        # TODO[5] : diffusion 8-connexes si self.p.diffuse_rate > 0
        self.pher = self.diffuse_pheromone()

    
    def diffuse_pheromone(self):
        if self.p.diffuse_rate <= 0:
            return self.pher
        else:
            avg_kernel = np.ones((3,3) , dtype = float)
            avg_kernel /= avg_kernel.sum()
            diffused = scipy.ndimage.convolve(self.pher , avg_kernel , mode = "constant" , cval = 0) # mode constant and cval 0 means that the pheromones seep out the edges of the world
            result = (1-self.p.diffuse_rate) * self.pher + self.p.diffuse_rate * diffused 
            #(1-diffuse rate) -> what exits from center ; diffuse_rate -> what enters into the neighbors and by continuity this of course sums to 1
            return result
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
        free = []
        n_neighbors = int(world.p.neighborhood)
        # x is col index # y is row index
        # positions are globally given as (x,y)
        if n_neighbors == 4:
            pos = [(self.x - 1 , self.y) , (self.x + 1 , self.y),
                   (self.x , self.y - 1) , (self.x , self.y + 1)] 
        elif n_neighbors == 8:
            pos = [(self.x - 1 , self.y) , (self.x -1  , self.y -1 ) , (self.x -1 , self.y + 1),
                   (self.x + 1 , self.y) , (self.x + 1  , self.y -1 ) , (self.x  +1 , self.y + 1),
                   (self.x , self.y +1) , (self.x , self.y  -1)]
        else:
            raise( ValueError(f"Unnacepted neighborhood number : {n_neighbors}"))

        for p in pos:
            fx , fy = p
            if world.in_bounds(fx , fy):
                if not world.blocked[fy , fx]:
                    free.append(p)

        return free

    def step(self, world: World) -> Tuple[int,int,float]:
        if self.has_food and not self.path_stack:
            # We expect to be on the nest now
            assert (self.x, self.y) == world.nest, \
                f"Backtracked to empty stack but not at nest: at {(self.x, self.y)}; nest {world.nest}"

        deposit = world.p.deposit_amount
        # TODO[7] : événements FOOD/NEST (changement de has_food, reset de path_stack)
        curr_cell =  world.grid[self.y , self.x]
        if curr_cell == CellType.FOOD and not self.has_food:
            self.has_food = True 
        elif curr_cell == CellType.NEST and self.has_food:
            self.has_food = False
            self.path_stack.clear()
        # TODO[8] : si retour et path_stack non vide -> backtracking + dépôt retour
        if self.has_food and self.path_stack:
            new_pos = self.path_stack.pop()
            deposit = world.p.deposit_amount_return
        # TODO[9] : sinon, choix epsilon-greedy (exploration) ou pondéré par (eps+pher)^alpha
        #           si has_food, multiplier les voisins qui rapprochent du nid par home_bias
        else:
            explore_random = (self.rng.random() <= world.p.explore_eps)
            free_neighbors = self.neighbors(world)
            if not free_neighbors:
                return(self.x , self.y , 0)
            # patch to prevent orbiting
            '''
            if not self.has_food:
                food_neighbors = [(cx, cy) for (cx, cy) in free_neighbors
                                if world.grid[cy, cx] == CellType.FOOD]
                if food_neighbors:
                    new_pos = self.rng.choice(food_neighbors)
                    self.path_stack.append((self.x, self.y))
                    self.x, self.y = new_pos
                    return self.x, self.y, world.p.deposit_amount
            '''
            if explore_random:
                idx = self.rng.randrange(len(free_neighbors))
                new_pos = free_neighbors[idx]
            else:
                self_nest_distance = abs(self.x - world.nest[0]) + abs(self.y - world.nest[1]) 
                eps = 1e-6
                weights = [(eps + world.pher[fy,fx])**world.p.alpha for fx,fy in free_neighbors]
                if self.has_food:
                    for i,(cx,cy) in enumerate(free_neighbors):
                        neighbor_nest_distance =  abs(cx - world.nest[0]) + abs(cy - world.nest[1])
                        if  neighbor_nest_distance < self_nest_distance:
                            weights[i] *= world.p.home_bias
                idx = self.rng.choices(range(len(free_neighbors)) , weights=weights,k=1)[0]
                new_pos = free_neighbors[idx]

        # TODO[10]: mise à jour position, mémoire du chemin (à l'aller), quantité déposée
        if not self.has_food:
            self.path_stack.append((self.x, self.y))
        
        self.x , self.y = new_pos
        
        return self.x , self.y , deposit
    

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
