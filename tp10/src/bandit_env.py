# bandit_env.py
import numpy as np

class KArmedBandit:
    """
    Bandit à K bras, chaque bras retournant une récompense entre 0 et 1
    qui évolue dans le temps.
    """
    def __init__(self, K: int, rng=None):
        self.K = K
        self.rng = np.random.default_rng() if rng is None else rng

        # Phase initiale
        phases = self.rng.uniform(0, 2*np.pi, size=K)
        self.phases = phases

    def true_probs(self, t: int):
        """
        Probabilités réelles à l'instant t.
        Variation sinusoïdale lente.
        """
        return 0.5 + 0.4 * np.sin(0.005 * t + self.phases)

    def step(self, action: int, t: int) -> float:
        """
        Joue le bras 'action' à l'instant t et renvoie la récompense (0 ou 1).
        """
        p = self.true_probs(t)[action]
        return float(self.rng.random() < p)
