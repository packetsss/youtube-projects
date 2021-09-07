import numpy as np

#======== UNIVERSAL ========#
FPS = 500
ZOOM_MULTIPLIER = 0.8
TOTAL_FOUL_TIMES = 25
WIDTH, HEIGHT = 1120, 620

TRAINING = True
DRAW_SCREEN = True # 140 fps --> 100 fps
USE_IMAGE_OBSERVATION = False
VELOCITY_LIMIT = int(420 * ZOOM_MULTIPLIER)

IMAGE_WIDTH = 120


#======== BALLS ========#
BALL_MASS = 10
BALL_FRICTION = 0.9
BALL_ELASTICITY = 0.95
BALL_DAMPING_THRESHOLD = 9
BALL_RADIUS = int(12 * ZOOM_MULTIPLIER)


#======== POCKETS ========#
if ZOOM_MULTIPLIER < 1:
    SIDE_POCKET_DIST = int(25 * ZOOM_MULTIPLIER)
    CORNER_POCKET_DIST = int(35 * ZOOM_MULTIPLIER)
else:
    SIDE_POCKET_DIST = int(32 * ZOOM_MULTIPLIER)
    CORNER_POCKET_DIST = int(42 * ZOOM_MULTIPLIER)
POCKET_RADIUS = int(20 * ZOOM_MULTIPLIER)

POCKET_LOCATION = (np.array([
    (CORNER_POCKET_DIST, CORNER_POCKET_DIST), 
    (WIDTH // 2, SIDE_POCKET_DIST), 
    (WIDTH - CORNER_POCKET_DIST, CORNER_POCKET_DIST), 
    (CORNER_POCKET_DIST, HEIGHT - CORNER_POCKET_DIST),
    (WIDTH // 2, HEIGHT - SIDE_POCKET_DIST),
    (WIDTH - CORNER_POCKET_DIST, HEIGHT - CORNER_POCKET_DIST)])
    * ZOOM_MULTIPLIER).astype(int)

HANGING_BALL_OFFSET = int(50 * ZOOM_MULTIPLIER)
HANGING_BALL_LOCATION = (np.array([
    (CORNER_POCKET_DIST + HANGING_BALL_OFFSET, CORNER_POCKET_DIST + HANGING_BALL_OFFSET), 
    (WIDTH // 2, SIDE_POCKET_DIST + HANGING_BALL_OFFSET), 
    (WIDTH - CORNER_POCKET_DIST - HANGING_BALL_OFFSET, CORNER_POCKET_DIST + HANGING_BALL_OFFSET), 
    (CORNER_POCKET_DIST + HANGING_BALL_OFFSET, HEIGHT - CORNER_POCKET_DIST - HANGING_BALL_OFFSET),
    (WIDTH // 2, HEIGHT - SIDE_POCKET_DIST - HANGING_BALL_OFFSET),
    (WIDTH - CORNER_POCKET_DIST - HANGING_BALL_OFFSET, HEIGHT - CORNER_POCKET_DIST - HANGING_BALL_OFFSET)])
    * ZOOM_MULTIPLIER).astype(int)

#======== RAILS ========#
RAIL_FRICTION = 0.5
RAIL_ELASTICITY = 0.8
RAIL_DISTANCE = int(54 * ZOOM_MULTIPLIER)
SIDE_RAIL_OFFSET = int(11.7 * ZOOM_MULTIPLIER)
SIDE_RAIL_OFFSET_2 = int(15 * ZOOM_MULTIPLIER)
CORNER_RAIL_OFFSET = int(31.7 * ZOOM_MULTIPLIER)

RAIL_LOCATION = (np.array([RAIL_DISTANCE, WIDTH - RAIL_DISTANCE, RAIL_DISTANCE, HEIGHT - RAIL_DISTANCE]) * ZOOM_MULTIPLIER).astype(int)

# top-left --> top-right --> bottom-right --> bottom-left
RAIL_POLY = [
    [[0, CORNER_RAIL_OFFSET], [RAIL_LOCATION[0], RAIL_LOCATION[2] + CORNER_RAIL_OFFSET], [RAIL_LOCATION[0], RAIL_LOCATION[3] - CORNER_RAIL_OFFSET], [0, int(HEIGHT * ZOOM_MULTIPLIER) - CORNER_RAIL_OFFSET]], # left
    [[CORNER_RAIL_OFFSET, 0], [int(WIDTH * ZOOM_MULTIPLIER / 2) - SIDE_RAIL_OFFSET_2, 0], [int(WIDTH * ZOOM_MULTIPLIER / 2) - SIDE_RAIL_OFFSET - SIDE_RAIL_OFFSET_2, RAIL_LOCATION[2]], [RAIL_LOCATION[0] + CORNER_RAIL_OFFSET, RAIL_LOCATION[2]]], # top 1
    [[int(WIDTH * ZOOM_MULTIPLIER / 2) + SIDE_RAIL_OFFSET_2, 0], [int(WIDTH * ZOOM_MULTIPLIER) - CORNER_RAIL_OFFSET, 0], [RAIL_LOCATION[1] - CORNER_RAIL_OFFSET, RAIL_LOCATION[2]], [int(WIDTH * ZOOM_MULTIPLIER / 2) + SIDE_RAIL_OFFSET + SIDE_RAIL_OFFSET_2, RAIL_LOCATION[2]]], # top 2
    [[RAIL_LOCATION[1], RAIL_LOCATION[2] + CORNER_RAIL_OFFSET], [int(WIDTH * ZOOM_MULTIPLIER), CORNER_RAIL_OFFSET], [int(WIDTH * ZOOM_MULTIPLIER), int(HEIGHT * ZOOM_MULTIPLIER) - CORNER_RAIL_OFFSET], [RAIL_LOCATION[1], RAIL_LOCATION[3] - CORNER_RAIL_OFFSET]], # right
    [[RAIL_LOCATION[0] + CORNER_RAIL_OFFSET, RAIL_LOCATION[3]], [int(WIDTH * ZOOM_MULTIPLIER / 2) - SIDE_RAIL_OFFSET - SIDE_RAIL_OFFSET_2, RAIL_LOCATION[3]], [int(WIDTH * ZOOM_MULTIPLIER / 2) - SIDE_RAIL_OFFSET_2, int(HEIGHT * ZOOM_MULTIPLIER)], [CORNER_RAIL_OFFSET, int(HEIGHT * ZOOM_MULTIPLIER)]], # bottom 1
    [[int(WIDTH * ZOOM_MULTIPLIER / 2) + SIDE_RAIL_OFFSET + SIDE_RAIL_OFFSET_2, RAIL_LOCATION[3]], [RAIL_LOCATION[1] - CORNER_RAIL_OFFSET, RAIL_LOCATION[3]], [int(WIDTH * ZOOM_MULTIPLIER) - CORNER_RAIL_OFFSET, int(HEIGHT * ZOOM_MULTIPLIER)], [int(WIDTH * ZOOM_MULTIPLIER / 2) + SIDE_RAIL_OFFSET_2, int(HEIGHT * ZOOM_MULTIPLIER)]] # bottom 2

]


#======== COLORS ========#
BLACK = (0, 0, 0)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)
RED = (255, 0, 0) 
GREEN = (0, 255, 0) 
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0) 
MAGENTA = (0, 255, 255) 
CYAN = (255, 0, 255)

TABLE_COLOR = (0, 100, 0)
TABLE_SIDE_COLOR = (200, 200, 0)
SOLIDS_COLOR = (250, 130, 0)
STRIPS_COLOR = (0, 110, 220)
