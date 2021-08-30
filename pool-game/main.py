from utils import *

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_VIDEO_WINDOW_POS"] = "300, 200"

import time
import random
import numpy as np
import pymunk as pm
import pygame as pg
import pymunk.pygame_util as pygame_util

class Pool:
    def __init__(self):
        self.space = pm.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.8

        # speed of the env
        self.dt = 1 / 10
        self.physics_steps_per_frame = 300

        # initialize pygame
        pg.init()
        self.screen = pg.display.set_mode((int(WIDTH * ZOOM_MULTIPLIER), int(HEIGHT * ZOOM_MULTIPLIER)))
        self.clock = pg.time.Clock()

        # render the game
        self.add_table()
        self.add_balls()
        self.draw_options = pygame_util.DrawOptions(self.screen)

        # initialize some constants
        self.steps_to_run_table = 0
        self.pocketed_balls = []
        self.running = True

        # 1 --> ball, 2 --> pocket, 3 --> rail
        self.collision_handler = self.space.add_collision_handler(1, 2)
        self.collision_handler.begin = self.ball_pocketed
        self.collision_handler.data["balls"] = self.balls

    def add_table(self):
        static_body = self.space.static_body
        self.rails = []
        for rail_poly in RAIL_POLY:
            rail = pm.Poly(static_body, rail_poly)
            rail.color = pg.Color(TABLE_SIDE_COLOR)
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
            # if overlap with another ball, choose another location
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

    def process_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_q:
                self.running = False
    
    @staticmethod
    def ball_pocketed(arbiter, space, data):
        # [ball, pocket]
        if arbiter.shapes[0].number != 0:
            data["balls"].remove(arbiter.shapes[0])
            space.remove(arbiter.shapes[0], arbiter.shapes[0].body)
        else:
            pm.Body.update_velocity(arbiter.shapes[0].body, (0, 0), damping=0, dt=1)
            arbiter.shapes[0].body.position = (500, 250)

        return False

    def redraw_screen(self):
        self.screen.fill(pg.Color(TABLE_COLOR))
        self.space.debug_draw(self.draw_options)

        pg.display.flip()
        self.clock.tick(FPS)
        pg.display.set_caption("FPS: " + str(self.clock.get_fps()))

    def run(self):
        while self.running:
            for _ in range(self.physics_steps_per_frame):
                balls_stopped = True
                for ball in self.balls:
                    if np.abs(ball.body.velocity).sum() < BALL_DAMPING_THRESHOLD:
                        pm.Body.update_velocity(ball.body, (0, 0), damping=0, dt=1)
                    else:
                        #print(ball.body.position, ball.body.velocity)
                        balls_stopped = False
                if balls_stopped:
                    velocity_range = 550
                    velocity = (random.randint(-velocity_range, velocity_range), random.randint(-velocity_range, velocity_range))
                    pm.Body.update_velocity(self.cue_ball.body, velocity, damping=0, dt=1)
                    break
                self.space.step(self.dt)

            if len(self.balls) < 2:
                print(f"Total steps: {self.steps_to_run_table}")
                self.steps_to_run_table = 0
                return

            self.process_events()
            self.redraw_screen()
            self.steps_to_run_table += 1


def main():
    pool = Pool()
    pool.run()

if __name__ == '__main__':
    main()