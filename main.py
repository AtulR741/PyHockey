import math
import pygame
from sys import exit
import random
import numpy as np

# Parameters
elasticity = 0.8
width = 400
height = 700
score = [0, 0]
time = 0
interaction = 0

DISC_WIDTH = 70
PUCK_WIDTH = 50
DISC_DISP = 10

FRAME_PENALTY = 0
FRAME_REWARD = 0.2
TIME_PENALTY = 5
GOAL_REWARD = 5
GOAL_PENALTY = 5
MISS_PENALTY = 3
TIMER = 1000

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def dist(x1, y1, x2, y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

class Disc():
    def __init__(self, w, h):
        self.x = 0
        self.y = 0
        self.prevX = 0
        self.prevY = 0
        self.width = w
        self.height = h
    def reset(self, x, y):
        self.x = x
        self.y = y
        self.prevX = x
        self.prevY = y
    def upperBoundary(self):
        return self.y < self.height * 0.5
    def lowerBoundary(self):
        return self.y + self.height * 0.5 > height
    def leftBoundary(self):
        return self.x < self.width * 0.5
    def rightBoundary(self):
        return self.x + self.width * 0.5 > width
    def hole(self):
        return (self.x >= (width*0.5 - self.width) and self.x <= (width*0.5 + self.width))

class User(Disc):
    def __init__(self, w, h):
        super().__init__(w, h)
        self.speedX = 0
        self.speedY = 0
    def reset(self, x, y):
        super().reset(x, y)
        self.speedX = 0
        self.speedY = 0
    def boundaryCheck(self):
        if super().rightBoundary():
            self.x = width - self.width * 0.5
        elif super().leftBoundary():
            self.x = self.width * 0.5
        if self.y - self.height * 0.5 < height * 0.5:
            self.y = (height + self.height) * 0.5
        elif super().lowerBoundary():
            self.y = height - self.height * 0.5
    def push(self, x, y):
        if y < 0.5*height:
            y = 0.5*height
        d = math.sqrt(dist(x, y, self.x, self.y))
        if d == 0:
            return
        speed = d/10
        self.speedX = speed*(x - self.x)/d
        self.speedY = speed*(y - self.y)/d
    def update(self):
        self.prevX = self.x
        self.prevY = self.y
        self.x += self.speedX
        self.y += self.speedY

class CPU(Disc):
    def boundaryCheck(self):
        if super().rightBoundary():
            self.x = width - self.width * 0.5
        elif super().leftBoundary():
            self.x = self.width * 0.5
        if self.y + self.height * 0.5 > height * 0.5:
            self.y = (height - self.height) * 0.5
        elif super().upperBoundary():
            self.y = self.height * 0.5
    def update(self, action):
        self.prevX = self.x
        self.prevY = self.y
        self.x += action[0]*2
        self.y += action[1]*2

class Puck(Disc):
    def __init__(self, w, h, ms):
        super().__init__(w, h)
        self.friction = 0.001
        self.speedX = 0
        self.speedY = 0
        self.maxSpeed = ms
    def reset(self, x, y):
        super().reset(x, y)
        self.speedX = 0
        self.speedY = 0
    def update(self):
        speed = math.sqrt(self.speedX * self.speedX + self.speedY * self.speedY)
        if speed == 0:
            return
        cos = abs(self.speedX / speed)
        sin = abs(self.speedY / speed)
        frictionX = self.friction * cos
        frictionY = self.friction * sin
        if self.speedX > 0:
            self.speedX = max(0, self.speedX - frictionX)
        elif self.speedX < 0:
            self.speedX = min(0, self.speedX + frictionX)
        if self.speedY > 0:
            self.speedY = max(0, self.speedY - frictionY)
        elif self.speedY < 0:
            self.speedY = min(0, self.speedY + frictionY)
        speed = math.sqrt(self.speedX * self.speedX + self.speedY * self.speedY)
        if speed > self.maxSpeed:
            if self.speedX != 0:
                self.speedX = (self.speedX) / abs(self.speedX) * self.maxSpeed * cos
            if self.speedY != 0:
                self.speedY = (self.speedY) / abs(self.speedY) * self.maxSpeed * sin
        self.x += self.speedX
        self.y += self.speedY
        
    def boundaryCheck(self):
        if super().rightBoundary():
            self.x = width - self.width * 0.5
            self.speedX *= -elasticity
        elif super().leftBoundary():
            self.x = self.width * 0.5
            self.speedX *= -elasticity
        if super().lowerBoundary():
            if super().hole():
                score[0] += 1
                return GOAL_REWARD
            self.y = height - self.height * 0.5
            self.speedY *= -elasticity
        elif super().upperBoundary():
            if super().hole():
                score[1] += 1
                return -GOAL_PENALTY
            self.y = self.height * 0.5
            self.speedY *= -elasticity
        return 0

epsilon = 1e-3

def collision(disc, puck):
    distance = dist(disc.x, disc.y, puck.x, puck.y)
    if distance * 4 >= (disc.width + puck.width) * (disc.width + puck.width) + epsilon:
        return -1

    normal = math.sqrt(dist(disc.x, disc.y, puck.x, puck.y))
    if normal == 0:
        normal = 1
    nx = abs((puck.x - disc.x) / normal)
    ny = abs((puck.y - disc.y) / normal)
    forceFactor = abs(puck.speedX - (disc.x - disc.prevX)) * abs(puck.speedY - (disc.y - disc.prevY)) * 8
    
    if puck.x > disc.x:
        disc.x -= DISC_DISP*nx
        puck.speedX += forceFactor * nx
    else:
        disc.x += DISC_DISP*nx
        puck.speedX -= forceFactor * nx
    if puck.y > disc.y:
        disc.y -= DISC_DISP*ny
        puck.speedY += forceFactor * ny
    else:
        disc.y += DISC_DISP*ny
        puck.speedY -= forceFactor * ny
    return distance

pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.Font('freesansbold.ttf', 30)

user = User(DISC_WIDTH, DISC_WIDTH)
cpu = CPU(DISC_WIDTH, DISC_WIDTH)
puck = Puck(PUCK_WIDTH, PUCK_WIDTH, 15)

def get_state():
        state = [
            (cpu.x - width*0.5)/width,
            cpu.y/height,
            (puck.x - cpu.x)/width,
            (puck.y - cpu.y)/height,
            puck.speedX/puck.maxSpeed,
            puck.speedY/puck.maxSpeed
        ]
        
        return np.array(state, dtype=np.float32)

userImage = pygame.transform.scale(pygame.image.load('User.png'), (DISC_WIDTH, DISC_WIDTH))
cpuImage = pygame.transform.scale(pygame.image.load('CPU.png'), (DISC_WIDTH, DISC_WIDTH))
userRect = userImage.get_rect()
cpuRect = cpuImage.get_rect()
puckImage = pygame.transform.scale(pygame.image.load('Puck.png'), (PUCK_WIDTH, PUCK_WIDTH))
puckRect = puckImage.get_rect()

def init(point):
    global time, interaction
    user.reset(width * 0.5, 0.8 * height)
    cpu.reset(width * 0.5, 0.2 * height)
    # puck.reset(width * 0.5, 0.7 * height)
    if point == 1:
        puck.reset(width * 0.5, 0.25 * height)
    else:
        puck.reset(width * 0.5, 0.75 * height)
    # puck.speedX = random.randint(-puck.maxSpeed, puck.maxSpeed)
    # puck.speedY = random.randint(-puck.maxSpeed, -2)
    time = 0
    interaction = 0

def gameInit(point):
    score[0] = 0
    score[1] = 0
    init(point)

def play_step(cpu_move):
    global time, interaction, FRAME_PENALTY
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:
        pos = pygame.mouse.get_pos()
        user.push(pos[0], pos[1])

    user.update()
    user.boundaryCheck()
    cpu.update(cpu_move)
    cpu.boundaryCheck()

    cpuReward, done, point = 0, False, 0

    # frame penalty
    # if puck.y < 0.5*height:
    #     cpuReward = -FRAME_PENALTY
    #     if sign(cpu_move[0] - puck.speedX) == sign(puck.x - cpu.x):
    #         cpuReward += FRAME_REWARD
    #     if sign(cpu_move[1] - puck.speedY) == sign(puck.y - cpu.y):
    #         cpuReward += FRAME_REWARD
    
    # miss penalty
    # if puck.y < cpu.y:
    #     cpuReward = -MISS_PENALTY

    # hit reward
    collision(user, puck)
    temp = collision(cpu, puck)
    # if temp != -1:
    #     cpuReward = max(3, temp*0.001)
    #     interaction = 2

    # check for delay
    # if puck.y <= 0.5*height:
    #     FRAME_PENALTY += 0.008
    #     if interaction == 0:
    #         interaction = 1
    #     time += 1
    # else:
    #     FRAME_PENALTY = 0
    #     time = 0
    #     if interaction == 1:
    #         cpuReward = -TIME_PENALTY
    #     interaction = 0
    # if time > TIMER:
    #     cpuReward = -TIME_PENALTY
    #     done = True
    # if puck.y > 0.5*height and puck.speedY == 0:
        # done = True

    puck.update()
    
    # goal reward
    temp = puck.boundaryCheck()
    if temp != 0:
        cpuReward = temp
        done = True
        point = sign(temp)

    screen.fill('BLACK')

    userRect.center = (user.x, user.y)
    screen.blit(userImage, userRect)
    cpuRect.center = (cpu.x, cpu.y)
    screen.blit(cpuImage, cpuRect)
    puckRect.center = (puck.x, puck.y)
    screen.blit(puckImage, puckRect)

    # if cpuReward != 0 and cpuReward != -FRAME_PENALTY and cpuReward != FRAME_REWARD:
    # if cpuReward:
    #     text = font.render(f'{cpuReward}', False, 'BLUE')
    #     textRect = text.get_rect(center = (40, height - 10))
    #     screen.blit(text, textRect)
    text = font.render(f'{score}', False, 'BLUE')
    textRect = text.get_rect(center = (width - 40, height - 10))
    screen.blit(text, textRect)

    pygame.display.update()
    clock.tick(60)

    return cpuReward, done, point