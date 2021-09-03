import numpy as np

#======== UNIVERSAL ========#
FPS = 500
VELOCITY_LIMIT = 500
ZOOM_MULTIPLIER = 1.2
BALL_DAMPING_THRESHOLD = 9
WIDTH, HEIGHT = 1120, 620
TOTAL_FOUL_TIMES = 25

TRAINING = True
DRAW_SCREEN = True # 140 fps --> 100 fps 

IMAGE_WIDTH = 120


#======== BALLS ========#
BALL_MASS = 10
BALL_FRICTION = 0.9
BALL_ELASTICITY = 0.95
BALL_RADIUS = int(12 * ZOOM_MULTIPLIER)


#======== POCKETS ========#
SIDE_POCKET_DIST = 40
CORNER_POCKET_DIST = 50
POCKET_RADIUS = int(20 * ZOOM_MULTIPLIER)

POCKET_LOCATION = (np.array([
    (CORNER_POCKET_DIST, CORNER_POCKET_DIST), 
    (WIDTH // 2, SIDE_POCKET_DIST), 
    (WIDTH - CORNER_POCKET_DIST, CORNER_POCKET_DIST), 
    (CORNER_POCKET_DIST, HEIGHT - CORNER_POCKET_DIST),
    (WIDTH // 2, HEIGHT - SIDE_POCKET_DIST),
    (WIDTH - CORNER_POCKET_DIST, HEIGHT - CORNER_POCKET_DIST)])
    * ZOOM_MULTIPLIER).astype(int)


#======== RAILS ========#
RAIL_FRICTION = 0.5
RAIL_ELASTICITY = 0.8
RAIL_DISTANCE = 65
SIDE_RAIL_OFFSET = 14
SIDE_RAIL_OFFSET_2 = 18
CORNER_RAIL_OFFSET = 38

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
