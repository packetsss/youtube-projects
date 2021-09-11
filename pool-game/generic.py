# Create by Packetsss
# Personal use is allowed
# Commercial use is prohibited

from pool import PoolEnv

import asyncio
import numpy as np
from copy import deepcopy


class Player:
    def __init__(self, env):
        self.env = env
        self.reward = -1
        self.low = self.env.action_space.low[0]
        self.high = self.env.action_space.high[0]
        self.v = np.random.uniform(self.low, self.high, 2)

    def clone(self):
        player = Player(self.env)
        player.v = np.copy(self.v)
        return player

    def revert_env(self, attrs):
        self.env.apply_attrs(attrs)

    def step(self):
        self.reward = self.env.step(self.v)[1]

    def mutate(self, mu=0.85):
        for i in range(len(self.v)):
            rand = np.random.uniform(0, 1)
            if rand < mu:
                if rand < mu / 5:
                    self.v = np.random.uniform(self.low, self.high, 2)
                else:
                    self.v[i] = min(
                        max(
                            self.v[i] + self.v[i] * np.random.normal(loc=0, scale=0.2),
                            self.low,
                        ),
                        self.high,
                    )


class GA:
    def __init__(self, env, population, iterations):
        self.env = env
        self.population = population
        self.iterations = iterations

        self.player = []
        self.best_player = Player(self.env)

    def train(self):
        iters = 0
        
        while 1:
            player = self.best_player.clone()
            player.mutate()

            attrs = player.env.copy_attrs()
            player.step()
            score_threshold = 1 if player.env.score_tracking["pot_count"] == 7 else 0.6
            player.revert_env(attrs)

            if self.best_player.reward < player.reward:
                self.best_player = player

            if iters == self.iterations or self.best_player.reward > score_threshold:
                return self.best_player, attrs
            iters += 1


population = 1
iterations = 500

pool = PoolEnv(training=True)

while 1:
    pool.training = True
    pool.draw_screen = False
    attrs = pool.copy_attrs()
    ga = GA(env=pool, population=population, iterations=iterations)
    player, attrs = ga.train()
    pool.apply_attrs(attrs)

    pool.training = False
    pool.draw_screen = True
    pool.dt = 20

    done = pool.step(player.v)[2]

    if done:
        pool.reset()
