#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ants import Params, Simulation


def make_richer_world(p: Params) -> Simulation:
    # nid au milieu
    nest = (p.width//2, p.height//2)
    # plusieurs sources de nourriture
    foods = [
        (p.width-12, p.height//2),
        (p.width-20, p.height//2 - 12),
        (p.width-20, p.height//2 + 12),
        (8, p.height//2)
    ]
    sim = Simulation(p, nest, foods)

    # Obstacles rectangulaires
    # Barrière centrale ouverte (un "portique")
    #sim.world.add_rect_obstacle(p.width//2 - 2, 0, 4, p.height//2 - 6)
    #sim.world.add_rect_obstacle(p.width//2 - 2, p.height//2 + 6, 4, p.height//2 - 6)

    # Deux blocs parasites
    #sim.world.add_rect_obstacle(p.width//3, p.height//3, 6, 10)
    #sim.world.add_rect_obstacle(2*p.width//3 - 8, 2*p.height//3 - 6, 10, 8)

    return sim


def main():
    p = Params(
        width=110, height=75, n_ants=600,
        evap_rate=0.012, diffuse_rate=0.06,
        deposit_amount=0.0,            # pas de dépôt à l'aller (optionnel)
        deposit_amount_return=2.0,     # dépôt fort au retour
        alpha=2.0, explore_eps=0.05, neighborhood="8", home_bias=1.6,
        seed=3
    )
    sim = make_richer_world(p)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    im = ax.imshow(sim.world.pher, cmap="magma", vmin=0.0, vmax=3.0,
                   origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Fourmis et phéromone")

    # Fourmilière + nourriture
    nx, ny = sim.world.nest
    foods_xy = np.array(sim.world.foods)
    ax.scatter([nx], [ny], c="cyan", s=70, marker="s", edgecolors="black", label="Fourmilière")
    ax.scatter(foods_xy[:,0], foods_xy[:,1], c="lime", s=80, marker="*", edgecolors="black", label="Nourriture")

    # Masque obstacles
    oy, ox = np.where(sim.world.blocked)
    ax.scatter(ox, oy, s=3, c="gray", alpha=0.6, label="Obstacles", marker='s')

    scat_ants = ax.scatter([], [], c="red", s=7, alpha=0.85, label="Fourmis")
    ax.legend(loc="upper left")

    def update(_frame):
        # plusieurs itération pour accélérer l'émergence
        for _ in range(5):
            sim.step()

        im.set_data(sim.world.pher)
        xs = [a.x for a in sim.ants]
        ys = [a.y for a in sim.ants]
        scat_ants.set_offsets(np.c_[xs, ys])
        ax.set_title(f"Fourmis et phéromone — itération {sim.t}")
        return (im, scat_ants)

    # IMPORTANT : conserver la référence à l'animation
    global _ANI_REF
    _ANI_REF = FuncAnimation(fig, update, frames=800, interval=25, blit=False)

    plt.show()


if __name__ == "__main__":
    main()
