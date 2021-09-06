from utils import *

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEO_WINDOW_POS"] = "50, 200"

import cv2
import time
import random
import numpy as np
import pymunk as pm
import pygame as pg
from gym import spaces
import pymunk.pygame_util as pygame_util

class PoolEnv:
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 500}
    def __init__(self):
        # initialize space
        self.space = pm.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.8
        self.space.collision_slop = 0.5
        self.space.idle_speed_threshold = 5
        self.space.sleep_time_threshold = 1e-8

        # gym environment
        self.spec = None
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, int(IMAGE_WIDTH * (HEIGHT / WIDTH)), IMAGE_WIDTH),
            dtype=np.uint8,
        )
        self.reward_range = np.array([0, 200])

        # speed of the env
        self.dt = 7
        self.step_size = 0.3
        if not TRAINING:
            self.step_size = 0.3

        # initialize pygame
        pg.init()
        self.screen = pg.display.set_mode((int(WIDTH * ZOOM_MULTIPLIER), int(HEIGHT * ZOOM_MULTIPLIER)))
        self.clock = pg.time.Clock()

        # render the game
        self.draw_options = pygame_util.DrawOptions(self.screen)

        # initialize some constants
        self.episodes = 0
        self.total_steps = 0
        self.running = True
        self.training = TRAINING
        self.draw_screen = DRAW_SCREEN
        self.total_foul_times = TOTAL_FOUL_TIMES

        # 1 --> ball, 2 --> pocket, 3 --> rail
        self.ball_collision_handler = self.space.add_collision_handler(1, 1)
        self.ball_collision_handler.begin = self.ball_contacted
        self.pocket_collision_handler = self.space.add_collision_handler(1, 2)
        self.pocket_collision_handler.begin = self.ball_pocketed
        self.rail_collision_handler = self.space.add_collision_handler(1, 3)
        self.rail_collision_handler.begin = self.rail_contacted
        self.reset()

    def add_table(self):
        static_body = self.space.static_body
        self.rails = []
        for rail_poly in RAIL_POLY:
            rail = pm.Poly(static_body, rail_poly)
            rail.color = pg.Color(TABLE_SIDE_COLOR)
            rail.collision_type = 3
            rail.elasticity = RAIL_ELASTICITY
            rail.friction = RAIL_FRICTION
            self.rails.append(rail)
        
        self.pockets = []
        for pocket_loc in POCKET_LOCATION:
            pocket = pm.Circle(static_body, POCKET_RADIUS, pocket_loc.tolist())
            pocket.color = pg.Color(BLACK)
            pocket.collision_type = 2
            pocket.elasticity = 0
            pocket.friction = 0
            self.pockets.append(pocket)
                    
        self.space.add(*self.rails, *self.pockets)
    
    def add_balls(self):
        # 0 -> cue-ball, 1-7 --> solids, 8 --> 8-ball, 9-15 -- strips
        self.balls = []
        positions = []
        for i in range(0, 16):
            intertia = pm.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, offset=(0, 0))
            ball_body = pm.Body(BALL_MASS, intertia)
            
            # initialize ball at random position
            ball_body.position = random.randint(RAIL_DISTANCE * 2, WIDTH * ZOOM_MULTIPLIER - RAIL_DISTANCE * 2), random.randint(RAIL_DISTANCE * 2, HEIGHT * ZOOM_MULTIPLIER - RAIL_DISTANCE * 2)

            # if overlap with another ball, choose a different location
            while 1:
                for pos in positions:
                    if distance_between_two_points(ball_body.position, pos) < BALL_RADIUS * 2:
                        break
                else:
                    break
                ball_body.position = random.randint(RAIL_DISTANCE * 2, WIDTH * ZOOM_MULTIPLIER - RAIL_DISTANCE * 2), random.randint(RAIL_DISTANCE * 2, HEIGHT * ZOOM_MULTIPLIER - RAIL_DISTANCE * 2)

            ball = pm.Circle(ball_body, BALL_RADIUS, offset=(0, 0))
            ball.elasticity = BALL_ELASTICITY
            ball.friction = BALL_FRICTION
            ball.collision_type = 1
            ball.number = i

            # separate ball types
            if i == 0:
                ball.color = pg.Color(WHITE)
                self.cue_ball = ball
            elif i < 8:
                ball.color = pg.Color(SOLIDS_COLOR)
            elif i == 8:
                ball.color = pg.Color(BLACK)
            else:
                ball.color = pg.Color(STRIPS_COLOR)
            
            positions.append(ball_body.position)
            self.balls.append(ball)
            self.space.add(ball, ball_body)

    @staticmethod
    def ball_contacted(arbiter, space, data):
        cb, bs = {data["cue_ball"]}, set(arbiter.shapes)

        if bs.issuperset(cb):
            if data["pocket_tracking"]["cue_ball_first_contact"] is None:
                data["pocket_tracking"]["cue_ball_first_contact"] = 1

            other_ball = bs.difference(cb).pop()
            if data["pocket_tracking"]["first_contacted_ball"] is None:
                data["pocket_tracking"]["first_contacted_ball"] = other_ball.number
        return True
    
    @staticmethod
    def ball_pocketed(arbiter, space, data):
        # arbiter: [ball, pocket]
        ball, pocket = arbiter.shapes
        data["potted_balls"].append(ball.number)

        if ball.number == 0:
            if data["pocket_tracking"]["cue_ball_first_contact"] is None:
                data["pocket_tracking"]["cue_ball_first_contact"] = 2
            data["pocket_tracking"]["cue_ball_pocketed"] = True
        elif ball.number == 8:
            # check for solids winning
            data["pocket_tracking"]["black_ball_pocketed"] = True
            if any([b.number >= 1 and b.number <= 7 for b in data["balls"]]):
                data["pocket_tracking"]["is_won"] = False
            else:
                data["pocket_tracking"]["is_won"] = True
        elif 1 <= ball.number <= 7:
            data["pocket_tracking"]["total_potted_balls"] += 1

        data["balls"].remove(ball)
        space.remove(ball, ball.body)
        return False

    @staticmethod
    def rail_contacted(arbiter, space, data):
        ball, rail = arbiter.shapes

        if ball.number == 0 and data["pocket_tracking"]["cue_ball_first_contact"] is None:
            data["pocket_tracking"]["cue_ball_first_contact"] = 3
        
        return True

    def redraw_screen(self):
        self.screen.fill(pg.Color(TABLE_COLOR))
        self.space.debug_draw(self.draw_options)

        pg.display.flip()
        self.clock.tick(FPS)

    def process_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_q:
                self.running = False
    
    def process_action(self, action):
        return (np.array(action) * VELOCITY_LIMIT).tolist()

    def process_observation(self):
        self.image = pg.surfarray.array3d(self.screen)
        self.image = cv2.rotate(self.image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.image = cv2.flip(self.image, 0)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (IMAGE_WIDTH, int(IMAGE_WIDTH * (HEIGHT / WIDTH))))
        self.image = self.image.reshape(3, int(IMAGE_WIDTH * (HEIGHT / WIDTH)), IMAGE_WIDTH)

    def step(self, action, *args, **kwargs):
        # waiting for all balls to stop
        action = self.process_action(action)
        self.cue_ball.body.activate()
        pm.Body.update_velocity(self.cue_ball.body, action, damping=0, dt=1)

        done = False
        info = {}
        self.reward = 5
        self.potted_balls.clear()
        self.pocket_tracking["cue_ball_pocketed"] = False
        self.pocket_tracking["first_contacted_ball"] = None
        self.pocket_tracking["cue_ball_first_contact"] = None
        while self.running:
            for ball in self.balls:
                if not ball.body.is_sleeping:
                    break
            else:
                break

            if not self.training:
                self.process_events()
                if self.draw_screen:
                    self.redraw_screen()
                self.space.step(self.step_size / self.dt)
            else:
                for _ in range(self.dt):
                    self.space.step(self.step_size / self.dt)

        fcb = self.pocket_tracking["first_contacted_ball"]
        # foul check
        if fcb is not None and 1 <= fcb <= 7:
            pck_arr = np.array(self.potted_balls)
            # pot ball check
            if any(pck_arr[(pck_arr >= 1) & (pck_arr <= 7)]):
                self.score_tracking["foul_count"] = 0
                self.score_tracking["touch_count"] += 1
                self.score_tracking["pot_count"] += 1
                self.reward += 40
            else:
                self.score_tracking["foul_count"] = 0
                self.score_tracking["touch_count"] += 1
                self.reward += 3
        else:
            self.score_tracking["foul_count"] += 1
            self.score_tracking["total_foul"] += 1
            self.score_tracking["touch_count"] = 0
            self.reward -= 3
        
        # if cue ball touch the rail first, subtract the reward
        if self.pocket_tracking["cue_ball_first_contact"] == 3:
            self.reward -= 2

        #self.reward += int(3 * (self.score_tracking["touch_count"] ** 1.2) - 3 * (self.score_tracking["foul_count"] ** 1.3))
        #self.reward += int(3 * (self.score_tracking["touch_count"] ** 1.2) - 3 * (self.score_tracking["foul_count"] ** 1.3))

        self.episode_reward.append(self.reward)
        # check endgame condition
        if self.pocket_tracking["black_ball_pocketed"] or self.score_tracking["total_foul"] > self.total_foul_times:
            self.episodes += 1
            done = True
            #self.reward = int(25 * (self.pocket_tracking["total_potted_balls"] ** 1.55) - 500)
            if self.pocket_tracking["is_won"] and not self.pocket_tracking["cue_ball_pocketed"]:
                #self.reward += 490
                self.reward = 200
                self.episode_reward.append(self.reward)

            pg.display.set_caption(f"FPS: {self.clock.get_fps():.0f}   TOT_REWARD: {np.sum(self.episode_reward):.0f}   POTTED_BALLS: {self.pocket_tracking['total_potted_balls']}   STEPS: {self.episode_steps}   TOTAL_STEPS: {self.total_steps}   EPISODES: {self.episodes}   ACTION: {np.array(action, dtype=int)}")
        
        if not self.training:
            pg.display.set_caption(f"FPS: {self.clock.get_fps():.0f}   REWARD: {self.reward:.0f}   POTTED_BALLS: {self.pocket_tracking['total_potted_balls']}   STEPS: {self.episode_steps}   TOTAL_STEPS: {self.total_steps}   EPISODES: {self.episodes}   ACTION: {np.array(action, dtype=int)}")

        # end the game if too many steps
        if self.episode_steps > 100:
            done = True
            self.reward = 0
        
        # prepare observation
        self.process_events()
        if self.draw_screen:
            self.redraw_screen()
        self.process_observation()
        self.episode_steps += 1
        self.total_steps += 1

        # adding cue ball back to table
        if self.pocket_tracking["cue_ball_pocketed"]:
            self.space.add(self.cue_ball.body, self.cue_ball)
            self.balls.append(self.cue_ball)

            pm.Body.update_velocity(self.cue_ball.body, (0, 0), damping=0, dt=1)
            self.cue_ball.body.position = random.randint(RAIL_DISTANCE * 2, WIDTH * ZOOM_MULTIPLIER - RAIL_DISTANCE * 2), random.randint(RAIL_DISTANCE * 2, HEIGHT * ZOOM_MULTIPLIER - RAIL_DISTANCE * 2)
            self.cue_ball.body.activate()

        return self.image, self.reward, done, info


    def reset(self, *args, **kwargs):
        for x in self.space.shapes:
            try:
                self.space.remove(x)
                self.space.remove(x.body)
            except AssertionError:
                pass

        self.add_table()
        self.add_balls()

        self.reward = 5 # self.total_foul_times * 5
        self.episode_reward = []
        self.potted_balls = []
        self.episode_steps = 0
        self.starting_time = time.time()
        self.score_tracking = {
            "foul_count": 0,
            "touch_count": 0,
            "pot_count": 0,
            "total_foul": 0
        }
        self.pocket_tracking = {
            "cue_ball_pocketed": False,
            "black_ball_pocketed": False,
            "is_won": None,
            "cue_ball_first_contact": None, # 1 --> ball, 2 --> pocket, 3 --> rail
            "first_contacted_ball": None,
            "total_potted_balls": 0,
        }

        self.ball_collision_handler.data["cue_ball"] = self.cue_ball
        self.ball_collision_handler.data["pocket_tracking"] = self.pocket_tracking

        self.pocket_collision_handler.data["balls"] = self.balls
        self.pocket_collision_handler.data["potted_balls"] = self.potted_balls
        self.pocket_collision_handler.data["pocket_tracking"] = self.pocket_tracking

        self.rail_collision_handler.data["pocket_tracking"] = self.pocket_tracking

        self.process_events()
        if self.draw_screen:
            self.redraw_screen()
        self.process_observation()

        return self.image

    def run(self):
        while self.running:
            velocity = (random.uniform(0, 1), random.uniform(0, 1))
            observation, reward, done, info = self.step(velocity)
            """ while 1:
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                        break
                else:
                    continue
                break"""
            if done:
                if self.pocket_tracking["is_won"] and not self.pocket_tracking["cue_ball_pocketed"]:
                    print("WIN!!")
                else:
                    print("LOSE!!")

                self.reset()


def main():
    pool = PoolEnv()
    pool.run()

if __name__ == '__main__':
    main()