import numpy as np

from object_3d import *


class Cube(Object3D):
    def __init__(self, renderer, shading):
        super().__init__(renderer, shading)

        self.vertexes = np.array([[-1, -1, -1, 1], [-1, 1, -1, 1], [1, 1, -1, 1], [1, -1, -1, 1],
                                  [-1, -1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, 1], [1, -1, 1, 1]])

        self.faces = np.array([[0, 1, 2, 3], [0, 1, 5, 4], [0, 4, 7, 3], [3, 2, 6, 7],
                               [1, 5, 6, 2], [4, 5, 6, 7]])

        self.scale(2)
        self.translate((5 , 0, 0))

        self.font = pg.font.SysFont('Arial', 30, bold=True)
        self.color_faces = [(pg.Color('blue'), face) for face in self.faces]
        self.movement_flag, self.draw_vertexes = True, False
        self.label = ''


class Tetrahedron(Object3D):
    def __init__(self, renderer, shading):
        super().__init__(renderer, shading)

        self.vertexes = np.array([[-1, -1, -1, 1], [1, -1, -1, 1], [0, -1, 1, 1], [0, 1, 0, 1]])
        self.faces = np.array([[0, 3, 1], [0, 2, 1], [0, 2, 3], [1, 2, 3]])
        self.color_faces = [(pg.Color('pink'), face) for face in self.faces]
        self.movement_flag, self.draw_vertexes = False, False
        self.scale(2)
        #self.translate((5, 5, 5))


class Axes(Object3D):
    def __init__(self, render, shading):
        super().__init__(render, shading)
        self.vertexes = np.array([(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
        self.faces = np.array([(0, 1), (0, 2), (0, 3)])
        self.colors = [pg.Color('red'), pg.Color('green'), pg.Color('blue')]
        self.color_faces = [(color, face) for color, face in zip(self.colors, self.faces)]
        self.draw_vertexes = False
        self.movement_flag = False
        self.label = 'XYZ'
