import pygame as pg
from vector import *


class Point():
    def __init__(self, render, pos: Vector2):
        self.render = render
        self.radius = 5
        self.pos = pos

    def draw(self):
        pg.draw.circle(self.render.screen, (255, 255, 255), (self.pos.x, self.pos.y), 5)

    def check_click(self, point2: Vector2):
        return point2.get_dist(point2) <= self.radius
