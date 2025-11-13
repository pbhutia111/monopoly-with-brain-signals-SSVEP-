from psychopy import visual, core, event
import numpy as np
from pyglet.window import key

# Colors in PsychoPy format (-1 to 1)
BLACK = (-1, -1, -1)
WHITE = (1, 1, 1)
ORANGE = (1, 0.294, -1)  # RGB(255, 165, 0) converted
PINK = (1, 0.506, 0.588)  # RGB(255, 192, 203) converted

WIDTH, HEIGHT = 1440, 900

# Create window
win = visual.Window(
    size=(WIDTH, HEIGHT),
    units='pix',
    color=BLACK,
    fullscr=True
)

FPS = 60
clock = core.Clock()

class Paddle:
    def __init__(self, posX, posY, width, height, speed, color):
        # Convert from top-left to center coordinates
        self.posX = posX + width / 2 - WIDTH / 2
        self.posY = HEIGHT / 2 - posY - height / 2
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color
        
        self.rect = visual.Rect(
            win=win,
            width=width,
            height=height,
            fillColor=color,
            lineColor=color,
            pos=(self.posX, self.posY)
        )
    
    def display(self):
        self.rect.draw()
    
    def update(self, dirX):
        self.posX += self.speed * dirX
        
        # Boundary checks (in PsychoPy coordinates)
        left_bound = -WIDTH / 2 + self.width / 2
        right_bound = WIDTH / 2 - self.width / 2
        
        if self.posX < left_bound:
            self.posX = left_bound
        if self.posX > right_bound:
            self.posX = right_bound
        
        self.rect.pos = (self.posX, self.posY)
    
    def getRect(self):
        return self.rect
    
    def getBounds(self):
        # Returns left, right, top, bottom
        return (
            self.posX - self.width / 2,
            self.posX + self.width / 2,
            self.posY + self.height / 2,
            self.posY - self.height / 2
        )


class Ball:
    def __init__(self, posX, posY, radius, speed, color):
        # Convert to center coordinates
        self.posX = posX - WIDTH / 2
        self.posY = HEIGHT / 2 - posY
        self.radius = radius
        self.speed = speed
        self.color = color
        self.dirX = 1
        self.dirY = 1
        self.count = 1
        self.default_speed = speed
        
        self.circle = visual.Circle(
            win=win,
            radius=radius,
            fillColor=color,
            lineColor=color,
            pos=(self.posX, self.posY)
        )
    
    def display(self):
        self.circle.draw()
    
    def update(self):
        self.posX += self.speed * self.dirX
        self.posY += self.speed * self.dirY
        
        # Side wall collisions
        if self.posX - self.radius <= -WIDTH / 2 or self.posX + self.radius >= WIDTH / 2:
            self.dirX *= -1
        
        # Top wall (top score)
        if self.posY + self.radius >= HEIGHT / 2 and self.count:
            self.count = 0
            return -1
        
        # Bottom wall (bottom score)
        elif self.posY - self.radius <= -HEIGHT / 2 and self.count:
            self.count = 0
            return 1
        
        self.circle.pos = (self.posX, self.posY)
        return 0
    
    def reset(self):
        self.posX = 0
        self.posY = 0
        self.dirY *= -1
        self.count = 1
        self.speed = self.default_speed
        self.circle.pos = (self.posX, self.posY)
    
    def hit(self):
        self.dirY *= -1
        self.speed += 0.5
        self.speed = min(self.speed, 10.0)
    
    def getBounds(self):
        return (
            self.posX - self.radius,
            self.posX + self.radius,
            self.posY + self.radius,
            self.posY - self.radius
        )


class FlickerBox:
    def __init__(self, posX, posY, width, height, frequency):
        self.width = width
        self.height = height
        self.frequency = frequency
        self.color = WHITE
        self.frame_count = 0
        self.frames_per_cycle = FPS / frequency
        
        # Convert to center coordinates
        self.posX = posX + width / 2 - WIDTH / 2
        self.posY = HEIGHT / 2 - posY - height / 2
        
        self.rect = visual.Rect(
            win=win,
            width=width,
            height=height,
            fillColor=WHITE,
            lineColor=WHITE,
            pos=(self.posX, self.posY)
        )
    
    def update(self, paddle_posX, paddle_posY, paddle_width, side):
        # Update position relative to paddle (in PsychoPy coordinates)
        if side == "left":
            self.posX = paddle_posX - paddle_width / 2 - self.width / 2 - 10
        else:  # right
            self.posX = paddle_posX + paddle_width / 2 + self.width / 2 + 10
        
        self.posY = paddle_posY
        self.rect.pos = (self.posX, self.posY)
    
    def display(self):
        self.frame_count += 1
        cycle_position = (self.frame_count % self.frames_per_cycle) / self.frames_per_cycle
        
        if cycle_position < 0.5:
            self.rect.fillColor = WHITE
            self.rect.lineColor = WHITE
        else:
            self.rect.fillColor = BLACK
            self.rect.lineColor = BLACK
        
        self.rect.draw()


def check_collision(ball, paddle):
    ball_left, ball_right, ball_top, ball_bottom = ball.getBounds()
    paddle_left, paddle_right, paddle_top, paddle_bottom = paddle.getBounds()
    
    return (ball_right >= paddle_left and 
            ball_left <= paddle_right and
            ball_top >= paddle_bottom and
            ball_bottom <= paddle_top)


def main():
    running = True
    
    top = Paddle(
        posX=WIDTH // 2 - 150,
        posY=20,
        width=300,
        height=15,
        speed=10,
        color=ORANGE
    )
    
    bottom = Paddle(
        posX=WIDTH // 2 - 150,
        posY=HEIGHT - 40,
        width=300,
        height=15,
        speed=10,
        color=PINK
    )
    
    ball = Ball(
        posX=WIDTH // 2,
        posY=HEIGHT // 2,
        radius=10,
        speed=4,
        color=WHITE
    )
    
    left_flicker = FlickerBox(
        posX=0,
        posY=0,
        width=50,
        height=50,
        frequency=10
    )
    
    right_flicker = FlickerBox(
        posX=0,
        posY=0,
        width=50,
        height=50,
        frequency=15
    )
    
    flicker_box = FlickerBox(
        posX=0,
        posY=0,
        width=2000,
        height=1000,
        frequency=15
    )
    
    
    
    paddles = [top, bottom]
    top_score = 0
    bottom_score = 0
    
    frame_start = core.getTime()
    
    # Access the pyglet keyboard state handler from PsychoPy window
    key_handler = key.KeyStateHandler()
    win.winHandle.push_handlers(key_handler)
    
    while running:
        # Check for escape key
        if key_handler[key.ESCAPE]:
            running = False
        
        # Top paddle (A/D) - check if keys are currently held down
        top_dirX = 0
        if key_handler[key.A]:
            top_dirX = -1
            print("A held")
        if key_handler[key.D]:
            top_dirX = 1
            print("D held")
        
        # Bottom paddle (LEFT/RIGHT)
        bottom_dirX = 0
        if key_handler[key.LEFT]:
            bottom_dirX = -1
            print("LEFT held")
        if key_handler[key.RIGHT]:
            bottom_dirX = 1
            print("RIGHT held")
        
        # Check collisions
        for paddle in paddles:
            if check_collision(ball, paddle):
                ball.hit()
        
        # Update positions
        top.update(top_dirX)
        bottom.update(bottom_dirX)
        
        left_flicker.update(bottom.posX, bottom.posY, bottom.width, "left")
        right_flicker.update(bottom.posX, bottom.posY, bottom.width, "right")
        
        point = ball.update()
        
        if point == -1:
            top_score += 1
            ball.reset()
        elif point == 1:
            bottom_score += 1
            ball.reset()
        
        # Draw everything
        top.display()
        bottom.display()
        ball.display()
        left_flicker.display()
        right_flicker.display()
        #flicker_box.display()
        
        win.flip()
        
        # Frame rate control
        elapsed = core.getTime() - frame_start
        if elapsed < 1.0 / FPS:
            core.wait(1.0 / FPS - elapsed)
        frame_start = core.getTime()
    
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
