from pool import PoolEnv

import numpy as np


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
        for i in range(len(self.v)):
            rand = np.random.uniform(0, 1)

            if rand < mu / 3:
                self.v = np.random.uniform(self.low, self.high, 2)

            elif rand < mu:
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

    def create_env(self):
        env = PoolEnv(training=True, draw_screen=False)
        env.apply_attrs(self.attrs)
        return env

    def train(self):
        for _ in range(self.iterations):
            player = self.best_player.clone()
            player.mutate()

            attrs = player.env.get_attrs()
            player.step()
            score_threshold = 1 if player.env.score_tracking["pot_count"] == 7 else 0.62
            player.env.apply_attrs(attrs)

            if self.best_player.reward < player.reward:
                self.best_player = player
            if self.best_player.reward > score_threshold:
                break

        return self.best_player, attrs


if __name__ == "__main__":
    iterations = 900
    pool = PoolEnv(training=True)

    while 1:
        pool.training = True
        pool.draw_screen = False

        attrs = pool.get_attrs()
        agent = GA(attrs, iterations=iterations)
        player, attrs = agent.train()
        pool.apply_attrs(attrs)

        pool.training = False
        pool.draw_screen = True

        done = pool.step(player.v)[2]

        if done:
            pool.reset()
