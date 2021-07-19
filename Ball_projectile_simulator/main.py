from utils import *

class Ball:
    def __init__(self, win, current_ball_pos):
        self.win = win
        self.pos = current_ball_pos

    def update(self, current_ball_pos, draw_ball=True):
        self.pos = current_ball_pos
        if draw_ball:
            self.draw_ball()

    def draw_ball(self):
        pg.draw.circle(surface=self.win, color=BLACK, center=self.pos, radius=BALL_RADIUS)
    
    def draw_line(self, mouse_pos):
        pg.draw.circle(surface=self.win, color=RED, center=mouse_pos, radius=2)
        pg.draw.line(surface=self.win, color=BLACK, start_pos=self.pos, end_pos=mouse_pos)

    def draw_ball_path(self, ball_loc_list):
        if ball_loc_list:
            for pos in ball_loc_list:
                pg.draw.circle(surface=self.win, color=RED, center=pos, radius=2)
    
    def calculate_angle(self, current_mouse_pos):
        x1, y1, x2, y2 = [*self.pos, *current_mouse_pos]

        return 90 if x1 - x2 == 0 else math.atan(abs(y1 - y2) / -(x1 - x2)) * 180 / math.pi

    
    def launch(self, time, current_mouse_pos, acceleration_time=ACCELERATION_MULTIPLIER):
        angle = self.calculate_angle(current_mouse_pos)
        angle = 180 + angle if angle < 0 else angle

        start_time = pg.time.get_ticks()
        acceleration = time * 100
        start_x_velocity = -acceleration_time * acceleration * math.cos(math.radians(angle))
        start_y_velocity = abs(acceleration_time * acceleration * math.sin(math.radians(angle)))
    
        return start_time, start_x_velocity, start_y_velocity

def main():
    win = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()

    current_ball_pos = INITIAL_BALL_POSITION
    ball = Ball(win, current_ball_pos)

    ball_path_list = []
    current_mouse_pos = None
    launch = False

    def exit():
        pg.display.quit()
        pg.quit()
        sys.exit()

    while 1:
        win.fill(WHITE)

        if launch:
            time_passed = (pg.time.get_ticks() - start_time) / 1000 * TIME_MULTIPLIER
            if current_ball_pos[1] <= INITIAL_BALL_POSITION[1]:
                x, y = start_ball_pos
                change_x = start_x_velocity * time_passed

                # Physics formula: final_pos = vi * time_passed + 1/2 * -GRAVITY * time_passed ** 2
                change_y = start_y_velocity * time_passed - 0.5 * GRAVITY * pow(time_passed, 2)

                current_ball_pos = (x - change_x, y - change_y)
                ball_path_list.append(current_ball_pos)
            else:
                launch = False
                current_ball_pos = (current_ball_pos[0], INITIAL_BALL_POSITION[1])

        for event in pg.event.get():
            if event.type == pg.QUIT:
                exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    launch = False
                    current_ball_pos = INITIAL_BALL_POSITION
            elif event.type == pg.MOUSEMOTION:
                current_mouse_pos = event.pos
            elif event.type == pg.MOUSEBUTTONDOWN:
                start_count_time = pg.time.get_ticks()
            elif event.type == pg.MOUSEBUTTONUP:
                end_time = (pg.time.get_ticks() - start_count_time) / 1000
                start_time, start_x_velocity, start_y_velocity = ball.launch(end_time, current_mouse_pos)
                start_ball_pos = current_ball_pos
                ball_path_list = []
                launch = True

        ball.update(current_ball_pos)
        if current_mouse_pos is not None:
            ball.draw_line(current_mouse_pos)
        ball.draw_ball_path(ball_path_list)

        clock.tick(FPS)
        pg.display.update()

if __name__ == '__main__':
    main()

    


