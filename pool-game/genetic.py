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


class GeneticAlgorithm:
    def __init__(self, attrs, iterations, training_type="mlp"):
        self.i = 0
        self.attrs = attrs
        self.iterations = iterations
        self.max_iterations = iterations * 5
        self.training_type = training_type
        self.best_player = Player(self.create_env())
        if self.training_type == "single":
            self.players = np.array([self.best_player])
            self.player_rewards = np.array([(self.best_player.reward + 1.001) ** 3])

    def create_env(self):
        env = PoolEnv(training=True, draw_screen=False)
        env.apply_attrs(self.attrs)
        return env

    def train(self, v=None, score_threshold=0.65):
        if v is not None:
            self.best_player.v = v
        for _ in range(self.iterations):
            if self.training_type == "single":
                if self.best_player.env.score_tracking["pot_count"] > 6:
                    player = self.best_player.clone()
                else:
                    player = np.random.choice(
                        a=self.players, p=self.player_rewards / self.player_rewards.sum()
                    ).clone()
            elif self.training_type == "mlp":
                player = self.best_player.clone()

            player.mutate()
            attrs = player.env.get_attrs()
            player.step()
            player.env.apply_attrs(attrs)

            if self.training_type == "single":
                self.players = np.append(self.players, player)
                self.player_rewards = np.append(
                    self.player_rewards, (player.reward + 2) ** 3
                )

            if self.best_player.reward < player.reward:
                self.best_player = player
            if self.best_player.reward > score_threshold:
                break
            self.i += 1
        
        #if self.best_player.reward < 0.5 and self.i < self.max_iterations:
        #    self.train(score_threshold=0.5)
            
        return self.best_player, attrs

class Trainer:
    def __init__(self, iterations, batch=10):
        self.iterations = iterations
        self.batch = batch
    
    def train(self, attrs):
        agent = GeneticAlgorithm(attrs, self.iterations, training_type="single")
        player, attrs = agent.train()
        return player.v, attrs
        
    def mlp_train(self, attrs):
        results = []
        self.itr = 0
        self.best_v = None
        with multiprocessing.Pool() as pooling:
            for i in range(self.batch):
                for result in pooling.imap_unordered(
                    self.mlp_get_player, itertools.repeat((attrs, self.iterations // self.batch), 10), chunksize=1
                ):  
                    if result[1] > 0.65 or (self.itr > self.iterations / 2 and result[1] > 0.5):
                        self.best_v, reward, itr, attrs = result
                        self.itr += itr
                        self.log_results(reward)
                        return self.best_v, attrs
                    results.append(result)
                else:
                    self.best_v, reward, itr, attrs = max(results, key=lambda x: x[1])
                    self.itr += itr
        self.log_results(reward)
        return self.best_v, attrs

    def log_results(self, reward):
        print(f"{self.itr} iterations, best reward {reward:.3f}")
        
    def mlp_get_player(self, x):
        attrs, iterations = x
        agent = GeneticAlgorithm(attrs, iterations=iterations, training_type="mlp")
        player, attrs = agent.train(self.best_v)
        return list(player.v), player.reward, agent.i, attrs


if __name__ == "__main__":
    iterations = 150
    pool = PoolEnv(training=False)
    trainer = Trainer(iterations)

    while 1:
        attrs = pool.get_attrs()
        best_player, attrs = trainer.mlp_train(attrs)

        pool.apply_attrs(attrs)
        done = pool.step(best_player)[2]

        if done:
            pool.reset()
