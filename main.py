from object_3d import *
from camera import *
from projection import *
import pygame as pg
from light import Light
from vector import Vector3
from objects_collection import Cube, Axes, Tetrahedron


class SoftwareRender:
    def __init__(self):
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 800, 600
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.create_objects()
        self.light = Light(Vector3(0, 20, -30))

    def create_objects(self):
        self.camera = Camera(self, [0, 6, -30], True)
        self.projection = Projection(self)
        # self.objects = [Cube(self, shading=True)]  # , Axes(self, shading=False)]  # self.get_object_from_file('resources/t_34_obj.obj')
        self.objects = [Tetrahedron(self, True, (0, 0, 5)), Cube(self, True, (0, 0, -5)), Cube(self, True, (5, 0, 0)), Cube(self, True, (-5, 0, 0))]

        # self.object.rotate_y(-math.pi / 4)
        # self.axes = Axes(self)

    # def get_object_from_file(self, filename):
    #     vertex, faces = [], []
    #     with open(filename) as f:
    #         for line in f:
    #             if line.startswith('v '):
    #                 vertex.append([float(i) for i in line.split()[1:]] + [1])
    #             elif line.startswith('f'):+
    #                 faces_ = line.split()[1:]
    #                 faces.append([int(face_.split('/')[0]) - 1 for face_ in faces_])
    #     return Object3D(self, vertex, faces)

    def draw(self):
        self.screen.fill(pg.Color('darkslategray'))
        # for object in self.objects:
        #     object.draw()
        Object3D.draw_objects(self, self.objects)
        # self.axes.draw()

    def run(self):
        while True:
            self.draw()
            self.camera.control()
            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.update()
            self.clock.tick(self.FPS)


if __name__ == '__main__':
    app = SoftwareRender()
    app.run()
