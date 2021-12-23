import numpy as np
import pygame
import matplotlib.pyplot as plt
pygame.init()

screen_width = 800
screen_height = 800
WHITE = (255,255,255)
BLACK = (0,0,0)
GREY = (128,128,128)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
DOT_SIZE = 8
screen = pygame.display.set_mode((screen_width,screen_height))
screen.fill(WHITE)



class Point():

    def __init__(self, x=None, y=None, size=None) -> None:
        if x == None:
            self.x = np.random.uniform(-1,1)#np.random.uniform(0,size[0])
        else:
            self.x = x
        if y == None:
            self.y = np.random.uniform(-1,1)#np.random.uniform(0,size[1])
        else:
            self.y = y
        self.px = self.map_range(self.x, -1, 1, 0, screen_width)
        self.py = self.map_range(self.y, -1, 1, screen_height, 0)
        self.bias = 1

        line_Y = f(self.x)

        if self.y > line_Y:
            self.label = 1
        else:
            self.label = -1

    def map_range(self, value, start1, stop1, start2, stop2):
        return (value - start1) / (stop1 - start1) * (stop2 - start2) + start2

    def show(self):
        if self.label == 1:
            pygame.draw.circle(screen, GREY, (self.px,self.py),DOT_SIZE)
            #plt.scatter(self.x, self.y,s=180, c="grey")
        else:
            pygame.draw.circle(screen, BLACK, (self.px,self.py),DOT_SIZE)
            #plt.scatter(self.x,self.y,s=180, c="black")

def f(x):
    # y = mx+b
    return 0.89*x-0.1




