from collections import defaultdict

import numpy as np

from state import State, action_array


class MCTS:
    def __init__(self, root: State, predictor):
        self.root = root
        self.predictor = predictor

        self.N = defaultdict(action_array)
        self.Q = defaultdict(action_array)
        self.P = dict()

    def _uct(self, k):
        return self.Q[k] + self.P[k] * np.sqrt(
            np.log(np.sum(self.N[k]) + 1) / (self.N[k] + 1)
        )

    def _playout(self, s: State):
        if s.is_terminal():
            return -s.reward

        key = s.hash()

        if key not in self.P:
            pi, r = self.predictor(s)
            self.P[key] = pi
            return -r

        uct = self._uct(key)
        a = np.argmax(uct)

        s.push(a)

        r = self._playout(s)
        self.Q[key][a] = (self.Q[key][a] * self.N[key][a] + r) / (self.N[key][a] + 1)
        self.N[key][a] += 1

        s.pop()

        return -r

    def run(self, n):
        for _ in range(n):
            self._playout(self.root)

    def policy(self):
        key = self.root.hash()

        if key not in self.N:
            return action_array()

        return self.N[key] / np.sum(self.N[key])
