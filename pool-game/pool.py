# Create by Packetsss
# Personal use is allowed
# Commercial use is prohibited


from utils import *

import os
import gym

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEO_WINDOW_POS"] = "50, 200"
gym.logger.set_level(40)

import cv2
import time
import random
import numpy as np
import pymunk as pm
import pygame as pg
from gym import spaces
from gym.utils import seeding
from copy import copy
import pymunk.pygame_util as pygame_util


class PoolEnv:
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 500}

    def __init__(
        self,
        training=TRAINING,
        num_balls=NUM_BALLS,
        draw_screen=DRAW_SCREEN,
        reward_by_steps=REWARD_BY_STEPS,
        total_foul_times=TOTAL_FOUL_TIMES,
        observation_type=OBSERVATION_TYPE,
    ):
        # initialize some constants
        self.episodes = 0
        self.total_steps = 0
        self.running = True

        self.width = WIDTH * ZOOM_MULTIPLIER
        self.height = HEIGHT * ZOOM_MULTIPLIER
        self.training = training
        self.num_balls = num_balls
        self.reward_by_steps = reward_by_steps
        self.total_foul_times = total_foul_times
        self.observation_type = observation_type
        self.draw_screen = draw_screen or (observation_type == "image") or not training

        # initialize space
        self.space = pm.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.83
        self.space.collision_slop = 0.5
        self.space.idle_speed_threshold = 5
        self.space.sleep_time_threshold = 1e-8

        # gym environment
        self.spec = None
        self.num_envs = 1
        self.reward_range = np.array([-1, 1])
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        if self.observation_type == "image":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH),
                dtype=np.uint8,
            )
        else:
            # ball_x, ball_y, ball_type(pocketed, solid, strips, 8-ball, cue-ball) x 16 balls
            self.table_info = np.concatenate(
                [np.array(RAIL_POLY).flatten(), POCKET_LOCATION.flatten()]
            )
            self.table_info = self.table_info / self.table_info.max()
            low = np.concatenate(
                [np.array([0, 0, 0] * self.num_balls), [0] * len(self.table_info)]
            )
            high = np.concatenate(
                [np.array([1, 1, 1] * self.num_balls), [1] * len(self.table_info)]
            )
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
            )

        # speed of the env
        self.dt = 24
        self.step_size = 0.45

        # initialize pygame
        if self.draw_screen:
            pg.init()
            self.screen = pg.display.set_mode((int(self.width), int(self.height)))
            self.clock = pg.time.Clock()
            self.draw_options = pygame_util.DrawOptions(self.screen)

        # 1 --> ball, 2 --> pocket, 3 --> rail
        self.ball_collision_handler = self.space.add_collision_handler(1, 1)
        self.ball_collision_handler.begin = self.ball_contacted
        self.pocket_collision_handler = self.space.add_collision_handler(1, 2)
        self.pocket_collision_handler.begin = self.ball_pocketed
        self.rail_collision_handler = self.space.add_collision_handler(1, 3)
        self.rail_collision_handler.begin = self.rail_contacted
        self.reset()

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    @property
    def unwrapped(self):
        return self

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        self.process_events()
        self.redraw_screen()

    def add_table(self):
        """
        filter: black ball, solids, cue-ball, pocket, rail
        --> 0b      0         0        0         0      0
        if change 0 to 1, means filter out the 1's
        e.g. 0b00111 means only account for unassigned and solids
        this is very complex and you won't understand hahaha
        """
        static_body = self.space.static_body
        self.rails = []
        for rail_poly in RAIL_POLY:
            rail = pm.Poly(static_body, rail_poly)
            rail.color = pg.Color(TABLE_SIDE_COLOR)
            rail.collision_type = 3
            rail.elasticity = RAIL_ELASTICITY
            rail.friction = RAIL_FRICTION
            rail.filter = pm.ShapeFilter(categories=0b00001)
            self.rails.append(rail)

        self.pockets = []
        for pocket_loc in POCKET_LOCATION:
            pocket = pm.Circle(static_body, POCKET_RADIUS, pocket_loc.tolist())
            pocket.color = pg.Color(BLACK)
            pocket.collision_type = 2
            pocket.elasticity = 0
            pocket.friction = 0
            pocket.filter = pm.ShapeFilter(categories=0b00010)
            self.pockets.append(pocket)

        self.space.add(*self.rails, *self.pockets)

    def add_balls(self):
        # 0 -> cue-ball, 1-7 --> solids, 8 --> 8-ball, 9-15 -- strips
        self.balls = []
        positions = []
        for i in range(self.num_balls):
            intertia = pm.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, offset=(0, 0))
            ball_body = pm.Body(BALL_MASS, intertia)

            # initialize ball at random position
            if self.num_balls < 2:
                ball_body.position = random.choice(HANGING_BALL_LOCATION).tolist()
            else:
                ball_body.position = random.randint(
                    RAIL_DISTANCE * 2, self.width - RAIL_DISTANCE * 2
                ), random.randint(RAIL_DISTANCE * 2, self.height - RAIL_DISTANCE * 2)

            # if overlap with another ball, choose a different location
            while 1:
                for pos in positions:
                    if (
                        distance_between_two_points(ball_body.position, pos)
                        < BALL_RADIUS * 2
                    ):
                        break
                else:
                    break
                if self.num_balls < 2:
                    ball_body.position = random.choice(HANGING_BALL_LOCATION).tolist()
                else:
                    ball_body.position = random.randint(
                        RAIL_DISTANCE * 2, self.width - RAIL_DISTANCE * 2
                    ), random.randint(
                        RAIL_DISTANCE * 2, self.height - RAIL_DISTANCE * 2
                    )

            ball = pm.Circle(ball_body, BALL_RADIUS, offset=(0, 0))
            ball.elasticity = BALL_ELASTICITY
            ball.friction = BALL_FRICTION
            ball.collision_type = 1
            ball.number = i
            ball.filter = pm.ShapeFilter(categories=0b00100)

            # separate ball types
            # observation_number: pocketed 0, solid 1, strips 2, 8-ball 3, cue-ball 4
            if i == 0:
                ball.color = pg.Color(WHITE)
                ball.observation_number = 4
                self.cue_ball = ball
            elif i < 8:
                ball.color = pg.Color(SOLIDS_COLOR)
                ball.observation_number = 1
                ball.filter = pm.ShapeFilter(categories=0b01000)
            elif i == 8:
                ball.color = pg.Color(BLACK)
                ball.observation_number = 3
                ball.filter = pm.ShapeFilter(categories=0b10000)
            else:
                ball.color = pg.Color(STRIPS_COLOR)
                ball.observation_number = 2

            positions.append(ball_body.position)
            self.balls.append(ball)
            self.space.add(ball, ball_body)

    @staticmethod
    def ball_contacted(arbiter, space, data):
        # count bank/carrom shot collisions
        if (
            data["pocket_tracking"]["first_contacted_ball"] in arbiter.shapes
            and not data["pocket_tracking"]["potted_balls"]
        ):
            data["pocket_tracking"]["fcb_collision_count"] += 1

        cb, bs = {data["cue_ball"]}, set(arbiter.shapes)
        if bs.issuperset(cb):
            other_ball = bs.difference(cb).pop()
            if data["pocket_tracking"]["first_contacted_ball"] is None:
                data["pocket_tracking"]["first_contacted_ball"] = other_ball
        return True

    @staticmethod
    def ball_pocketed(arbiter, space, data):
        # arbiter: [ball, pocket]
        ball, pocket = arbiter.shapes
        data["pocket_tracking"]["potted_balls"].append(ball.number)

        if ball.number == 0:
            data["pocket_tracking"]["cue_ball_pocketed"] = True
        elif ball.number == 8:
            # check for solids winning
            data["pocket_tracking"]["black_ball_pocketed"] = True
            if (
                any([b.number >= 1 and b.number <= 7 for b in data["balls"]])
                or data["pocket_tracking"]["first_contacted_ball"].number != 8
            ):
                data["pocket_tracking"]["is_won"] = False
            else:
                data["pocket_tracking"]["is_won"] = True
        elif 1 <= ball.number <= 7:
            data["pocket_tracking"]["total_potted_balls"] += 1

        try:
            data["balls"].remove(ball)
        except ValueError:
            pass
        space.remove(ball, ball.body)
        return False

    @staticmethod
    def rail_contacted(arbiter, space, data):
        ball, rail = arbiter.shapes
        # count bank/carrom shot collisions
        if (
            data["pocket_tracking"]["first_contacted_ball"] == ball
            and not data["pocket_tracking"]["potted_balls"]
        ):
            data["pocket_tracking"]["fcb_collision_count"] += 1

        if ball.number == 0 and data["pocket_tracking"]["first_contacted_ball"] is None:
            data["pocket_tracking"]["rail_collision_count"] += 1

        return True

    def redraw_screen(self):
        self.screen.fill(pg.Color(TABLE_COLOR))
        self.space.debug_draw(self.draw_options)

        pg.display.flip()
        self.clock.tick(FPS)

    def process_events(self):
        for event in pg.event.get():
            if (
                event.type == pg.QUIT
                or event.type == pg.KEYDOWN
                and event.key == pg.K_q
            ):
                self.running = False

    def process_action(self, action):
        return (np.array(action) * VELOCITY_LIMIT).tolist()

    def process_observation(self):
        if self.observation_type == "image":
            img = pg.surfarray.pixels3d(self.screen)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.flip(img, 0)

            # cv2.imshow("", cv2.resize(img, (300, 300)))
            # if cv2.waitKey(1) == ord("q"):
            #    os._exit(0)
            return img.reshape(3, IMAGE_HEIGHT, IMAGE_WIDTH)
        elif self.observation_type == "vector":
            # observation_number: pocketed 0, solid 1, strips 2, 8-ball 3, cue-ball 4
            width_multiplier = (IMAGE_WIDTH / (self.width)) / IMAGE_WIDTH
            height_multiplier = (IMAGE_HEIGHT / (self.height)) / IMAGE_HEIGHT
            obs = np.array(
                [
                    np.array(
                        [
                            x.body.position[0] * width_multiplier,
                            x.body.position[1] * height_multiplier,
                            x.observation_number / 4,
                        ]
                    )
                    for x in self.balls
                ]
            )

            balls_to_fill = self.num_balls - len(self.balls)
            if len(self.balls) == 0:
                # if no balls on table
                obs = (
                    np.repeat(np.array([0, 0, 0]), balls_to_fill, axis=0)
                    .reshape(balls_to_fill, 3)
                    .flatten()
                )
            if balls_to_fill > 0:
                # if some balls are pocketed
                obs = np.vstack(
                    (
                        obs,
                        np.repeat(np.array([0, 0, 0]), balls_to_fill, axis=0).reshape(
                            balls_to_fill, 3
                        ),
                    )
                ).flatten()
            else:
                obs = obs.flatten()
            obs = np.concatenate([obs, self.table_info])
            return obs
        else:
            return

    def process_reward(self, reward=None):
        # normalizing the reward to range(-1, 1)
        if reward is None:
            reward = self.reward
        if self.reward_by_steps:
            return np.clip((reward / 50) * 2 - 1, -1, 1)
        else:
            return np.clip((reward / 500) * 2 - 1, -1, 1)

    def step(self, action, *args, **kwargs):
        # waiting for all balls to stop
        action = self.process_action(action)
        self.cue_ball.body.activate()
        pm.Body.update_velocity(self.cue_ball.body, action, damping=0, dt=1)

        # reset some constants
        done = False
        info = {}
        if self.reward_by_steps:
            self.reward = 7
        closest_ball_dist = 1e6
        closest_pocket_dist = 1e6
        self.pocket_tracking["potted_balls"].clear()
        self.pocket_tracking["fcb_collision_count"] = 0
        self.pocket_tracking["rail_collision_count"] = 0
        self.pocket_tracking["cue_ball_pocketed"] = False
        self.pocket_tracking["first_contacted_ball"] = None
        while self.running:
            # check if all balls stopped
            for ball in self.balls:
                if not ball.body.is_sleeping:
                    break
            else:
                break

            # reward for how close cue_ball to ball and ball to pocket
            fcb = self.pocket_tracking["first_contacted_ball"]
            if self.score_tracking["pot_count"] == 7:
                if fcb is None:
                    bl = self.space.point_query_nearest(
                        tuple(self.cue_ball.body.position),
                        self.width * 0.8,
                        pm.ShapeFilter(mask=pm.ShapeFilter.ALL_MASKS() ^ 0b01111),
                    )
                    if bl is not None:
                        closest_ball_dist = min(bl.distance, closest_ball_dist)
                elif fcb.number == 8:
                    poc = self.space.point_query_nearest(
                        tuple(fcb.body.position),
                        self.width * 0.5,
                        pm.ShapeFilter(mask=pm.ShapeFilter.ALL_MASKS() ^ 0b11101),
                    )
                    if poc is not None:
                        closest_pocket_dist = min(poc.distance, closest_pocket_dist)
            elif fcb is None:
                # filter out everything but balls
                bl = self.space.point_query_nearest(
                    tuple(self.cue_ball.body.position),
                    self.width * 0.8,
                    pm.ShapeFilter(mask=pm.ShapeFilter.ALL_MASKS() ^ 0b10111),
                )
                if bl is not None:
                    closest_ball_dist = min(bl.distance, closest_ball_dist)
            elif 1 <= fcb.number <= 7:
                # filter out everything but pockets
                poc = self.space.point_query_nearest(
                    tuple(fcb.body.position),
                    self.width * 0.5,
                    pm.ShapeFilter(mask=pm.ShapeFilter.ALL_MASKS() ^ 0b11101),
                )
                if poc is not None:
                    closest_pocket_dist = min(poc.distance, closest_pocket_dist)

            # step through
            if not self.training:
                self.process_events()
                self.redraw_screen()
                self.space.step(self.step_size / self.dt)
            else:
                for _ in range(self.dt):
                    self.space.step(self.step_size / self.dt)

        # rewarding

        # if cue ball touch the rail first, subtract the reward
        self.reward -= 2 * self.pocket_tracking["rail_collision_count"]

        fcb = self.pocket_tracking["first_contacted_ball"]
        if (
            fcb is not None
            and not self.pocket_tracking["cue_ball_pocketed"]
            and (
                1 <= fcb.number <= 7
                or (self.score_tracking["pot_count"] == 7 and fcb.number == 8)
            )
        ):
            pck_arr = np.array(self.pocket_tracking["potted_balls"])
            if any(pck_arr[(pck_arr >= 1) & (pck_arr <= 8)]):
                # potted the correct ball
                self.score_tracking["foul_count"] = 0
                self.score_tracking["touch_count"] += 1
                self.score_tracking["pot_count"] += 1
                if self.reward_by_steps:
                    self.reward += 35
                else:
                    self.reward += 40

                self.reward -= self.pocket_tracking["fcb_collision_count"]
            else:
                # touched the correct ball
                # although didn't pot ball, still reward by how close ball gets to the pocket
                self.score_tracking["foul_count"] = 0
                self.score_tracking["touch_count"] += 1
                self.reward += 5
                if closest_pocket_dist < 600:
                    self.reward += 55 / np.sqrt(max(closest_pocket_dist, 1e-6))
        else:
            # touched the wrong ball or not touching anything at all
            # although fouled, still reward by how close it gets to the ball
            self.score_tracking["foul_count"] += 1
            self.score_tracking["total_foul"] += 1
            self.score_tracking["touch_count"] = 0
            self.reward -= 5
            if closest_ball_dist < 600:
                self.reward += 25 / np.sqrt(max(closest_ball_dist, 1e-6))

        # total episode reward
        if self.reward_by_steps:
            self.episode_reward.append(self.process_reward())

        # only cue ball left when use less balls
        if self.num_balls < 9 and len(self.balls) < 2 and self.cue_ball in self.balls:
            self.pocket_tracking["black_ball_pocketed"] = True
            self.pocket_tracking["is_won"] = True

        # check endgame condition
        if self.pocket_tracking["black_ball_pocketed"] or self.score_tracking[
            "total_foul"
        ] > self.total_foul_times / (self.score_tracking["pot_count"] * 0.2 + 1):
            self.episodes += 1
            done = True
            if (
                self.pocket_tracking["is_won"]
                and not self.pocket_tracking["cue_ball_pocketed"]
            ):
                if self.observation_type != "none":
                    if self.reward_by_steps:
                        self.reward = 50
                        self.episode_reward.append(self.process_reward())
                    else:
                        self.reward = 500
            else:
                self.reward = 0

            # update caption every 5 episodes
            if self.episodes == 1 or self.episodes % 5 == 0:
                self.fps = 1 / (
                    (time.time() - self.starting_time) / (self.episode_steps + 1e-8)
                )
                if not self.reward_by_steps:
                    pg.display.set_caption(
                        f"FPS: {self.fps:.0f}   REWARD: {self.process_reward():.3f}   POTTED_BALLS: {self.pocket_tracking['total_potted_balls']}   STEPS: {self.episode_steps}   TOTAL_STEPS: {self.total_steps}   EPISODES: {self.episodes}   ACTION: {np.array(action, dtype=int)}"
                    )
                else:
                    pg.display.set_caption(
                        f"FPS: {self.fps:.0f}   TOT_REWARD: {np.sum(self.episode_reward):.2f}   POTTED_BALLS: {self.pocket_tracking['total_potted_balls']}   STEPS: {self.episode_steps}   TOTAL_STEPS: {self.total_steps}   EPISODES: {self.episodes}   ACTION: {np.array(action, dtype=int)}"
                    )

        if not self.training:
            pg.display.set_caption(
                f"FPS: {self.clock.get_fps():.0f}   REWARD: {self.process_reward():.3f}   POTTED_BALLS: {self.pocket_tracking['total_potted_balls']}   STEPS: {self.episode_steps}   TOTAL_STEPS: {self.total_steps}   EPISODES: {self.episodes}   ACTION: {np.array(action, dtype=int).tolist()}"
            )

        # prepare observation
        if self.draw_screen:
            self.process_events()
            self.redraw_screen()

        self.episode_steps += 1
        self.total_steps += 1

        # adding cue ball back to table if potted
        if self.pocket_tracking["cue_ball_pocketed"]:
            self.space.add(self.cue_ball.body, self.cue_ball)
            self.balls.append(self.cue_ball)

            pm.Body.update_velocity(self.cue_ball.body, (0, 0), damping=0, dt=1)
            self.cue_ball.body.position = random.randint(
                RAIL_DISTANCE * 2, self.width - RAIL_DISTANCE * 2
            ), random.randint(RAIL_DISTANCE * 2, self.height - RAIL_DISTANCE * 2)
            self.cue_ball.body.activate()

        return self.process_observation(), self.process_reward(), done, info

    def reset(self, *args, **kwargs):
        for x in self.space.shapes:
            try:
                self.space.remove(x)
                self.space.remove(x.body)
            except AssertionError:
                pass

        self.add_table()
        self.add_balls()

        self.reward = self.total_foul_times * 5
        self.episode_reward = []
        self.episode_steps = 0
        self.starting_time = time.time()
        self.score_tracking = {
            "foul_count": 0,
            "touch_count": 0,
            "pot_count": 0,
            "total_foul": 0,
        }
        self.pocket_tracking = {
            "cue_ball_pocketed": False,
            "black_ball_pocketed": False,
            "is_won": None,
            "cue_ball_first_contact": None,  # 1 --> ball, 2 --> pocket, 3 --> rail
            "first_contacted_ball": None,
            "total_potted_balls": 0,
            "fcb_collision_count": 0,
            "rail_collision_count": 0,
            "potted_balls": [],
        }

        self.ball_collision_handler.data["cue_ball"] = self.cue_ball
        self.ball_collision_handler.data["pocket_tracking"] = self.pocket_tracking
        self.pocket_collision_handler.data["balls"] = self.balls
        self.pocket_collision_handler.data["pocket_tracking"] = self.pocket_tracking
        self.rail_collision_handler.data["pocket_tracking"] = self.pocket_tracking

        if self.draw_screen:
            self.process_events()
            self.redraw_screen()
        return self.process_observation()

    def get_attrs(self):
        def get_balls(x):
            return {
                "position": list(x.body.position),
                "number": x.number,
                "color": x.color,
                # "filter": x.filter,
                "collision_type": x.collision_type,
                "observation_number": x.observation_number,
            }

        return {
            "balls_attrs": [get_balls(x) for x in self.balls],
            "reward": copy(self.reward),
            "episodes": copy(self.episodes),
            "episode_reward": copy(self.episode_reward),
            "episode_steps": copy(self.episode_steps),
            "total_steps": copy(self.total_steps),
            "starting_time": copy(self.starting_time),
            "score_tracking": copy(self.score_tracking),
            "pocket_tracking": copy(self.pocket_tracking),
        }

    def apply_attrs(self, attrs):
        balls_attrs = attrs["balls_attrs"]
        self.reward = attrs["reward"]
        self.episodes = attrs["episodes"]
        self.episode_reward = attrs["episode_reward"]
        self.episode_steps = attrs["episode_steps"]
        self.total_steps = attrs["total_steps"]
        self.starting_time = attrs["starting_time"]
        self.score_tracking = attrs["score_tracking"]
        self.pocket_tracking = attrs["pocket_tracking"]
        del attrs

        balls = []
        for i in range(len(balls_attrs)):
            if i < len(self.balls):
                self.space.remove(self.balls[i].body, self.balls[i])
            intertia = pm.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, offset=(0, 0))
            ball_body = pm.Body(BALL_MASS, intertia)
            ball_body.position = balls_attrs[i]["position"]
            ball = pm.Circle(ball_body, BALL_RADIUS, offset=(0, 0))
            ball.elasticity = BALL_ELASTICITY
            ball.friction = BALL_FRICTION
            ball.collision_type = balls_attrs[i]["collision_type"]
            ball.number = balls_attrs[i]["number"]
            ball.color = balls_attrs[i]["color"]
            ball.observation_number = balls_attrs[i]["observation_number"]
            # ball.filter = balls_attrs[i]["filter"]
            ball.filter = pm.ShapeFilter(categories=0b00100)
            if ball.number == 0:
                self.cue_ball = ball
            elif ball.number < 8:
                ball.filter = pm.ShapeFilter(categories=0b01000)
            elif ball.number == 8:
                ball.filter = pm.ShapeFilter(categories=0b10000)
            else:
                ball.filter = pm.ShapeFilter(categories=0b00100)
            self.space.add(ball_body, ball)
            balls.append(ball)
        self.balls = balls

        self.ball_collision_handler.data["cue_ball"] = self.cue_ball
        self.ball_collision_handler.data["pocket_tracking"] = self.pocket_tracking
        self.pocket_collision_handler.data["balls"] = self.balls
        self.pocket_collision_handler.data["pocket_tracking"] = self.pocket_tracking
        self.rail_collision_handler.data["pocket_tracking"] = self.pocket_tracking

    def run(self, model=None):
        observation = self.reset()
        while self.running:
            if model is not None:
                velocity = model.predict(observation)[0].tolist()
            else:
                velocity = (random.uniform(-1, 1), random.uniform(-1, 1))
            observation, reward, done, info = self.step(velocity)
            # whether use space to move through each step
            """ while 1:
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                        break
                else:
                    continue
                break"""
            if done:
                if not self.training:
                    if (
                        self.pocket_tracking["is_won"]
                        and not self.pocket_tracking["cue_ball_pocketed"]
                    ):
                        print("WIN !!")
                    else:
                        print("LOSE!!")
                self.reset()

    def close(self):
        pg.quit()


def main():
    import pstats
    import pathlib
    import cProfile

    pool = PoolEnv(training=False, draw_screen=True, observation_type="vector")

    # with cProfile.Profile() as pr:
    pool.run()

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats("profiling.prof")


if __name__ == "__main__":
    main()
