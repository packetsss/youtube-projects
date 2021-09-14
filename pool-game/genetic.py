import sys
from pool import PoolEnv

from concurrent.futures import ProcessPoolExecutor
import numpy as np

import multiprocessing
import itertools


class Player:
    def __init__(self, env: PoolEnv) -> None:
        self.env = env
        self.reward = -1
        self.low = self.env.action_space.low[0]
        self.high = self.env.action_space.high[0]
        self.v = np.random.uniform(low=self.low, high=self.high, size=2)

    def clone(self):
        player = Player(self.env)
        player.v = np.copy(self.v)
        return player

    def step(self):
        self.reward = self.env.step(self.v)[1]

    def mutate(self, mu=1):
        rand = np.random.uniform(0, 1)
        if rand < mu / 5:
            self.v = np.random.uniform(self.low, self.high, 2)
        elif rand < mu:
            for i in range(len(self.v)):
                self.v[i] = min(
                    max(
                        self.v[i] + self.v[i] * np.random.normal(loc=0, scale=0.3),
                        self.low,
                    ),
                    self.high,
                )


class GA:
    def __init__(self, attrs, iterations):
        self.attrs = attrs
        self.iterations = iterations
        self.best_player = Player(self.create_env())
        self.players = np.array([self.best_player])
        self.player_rewards = np.array([(self.best_player.reward + 1.001) ** 3])

    def create_env(self):
        env = PoolEnv(training=True, draw_screen=False)
        env.apply_attrs(self.attrs)
        return env

    def train(self):
        for _ in range(self.iterations):
            # if self.best_player.env.score_tracking["pot_count"] > 6:
            #    player = self.best_player.clone()
            # else:
            #    player = np.random.choice(
            #        a=self.players, p=self.player_rewards / self.player_rewards.sum()
            #    ).clone()
            player = self.best_player.clone()

            player.mutate()

            attrs = player.env.get_attrs()
            player.step()
            score_threshold = 1 if player.env.score_tracking["pot_count"] == 7 else 0.65
            player.env.apply_attrs(attrs)

            self.players = np.append(self.players, player)
            self.player_rewards = np.append(
                self.player_rewards, (player.reward + 2) ** 3
            )

            if self.best_player.reward < player.reward:
                self.best_player = player
            if self.best_player.reward > score_threshold:
                break

        return self.best_player, attrs


def get_player(x):
    attrs, iterations = x
    agent = GA(attrs, iterations=iterations)
    player, attrs = agent.train()
    print("finished")
    return list(player.v), player.reward, attrs


if __name__ == "__main__":
    pool = PoolEnv(training=False)

    while 1:
        iterations = 120
        attrs = pool.get_attrs()
        # agent = GA(attrs, iterations=iterations)
        # best_player, attrs = agent.train()
        with multiprocessing.Pool() as pooling:
            results = []
            for result in pooling.imap_unordered(
                get_player, itertools.repeat((attrs, iterations), 10), chunksize=1
            ):
                if result[1] > 0.65:
                    print(111)
                    best_player, reward, attrs = result
                    break
                results.append(result)
            else:
                best_player, reward, attrs = max(results, key=lambda x: x[1])

            print("reward: ", reward)

        pool.apply_attrs(attrs)

        try:
            done = pool.step(best_player)[2]
        except KeyboardInterrupt:
            sys.exit("Inter")

        if done:
            pool.reset()
