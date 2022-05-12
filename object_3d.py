import pygame as pg
from matrix_functions import *
from numba import njit
from vector import *

dim = 0.01


@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))


class Object3D:

    def __init__(self, render, shading=True, vertexes='', faces=''):
        self.render = render
        self.vertexes = np.array([np.array(v) for v in vertexes])
        self.faces = np.array([np.array(face) for face in faces])
        self.shading = shading

        self.font = pg.font.SysFont('Arial', 30, bold=True)
        self.color_faces = [(pg.Color('orange'), face) for face in self.faces]
        self.movement_flag, self.draw_vertexes = True, False
        self.label = ''

    # @classmethod
    # def draw_objects(cls, objects: list):
    #
    #
    #     for object in objects:
    #         vertexes, colors = object.screen_projection()

    def draw(self):
        self.screen_projection()
        self.movement()

    def movement(self):
        if self.movement_flag:
            self.rotate_y(-(pg.time.get_ticks() % 0.005))
        pass

    def screen_projection(self):

        shaded_colors = [(self.calculate_shade(face_color[1], face_color[0]), face_color[1]) for face_color in
                         self.color_faces]

        vertexes = self.vertexes @ self.render.camera.camera_matrix()  # переводим в пространство камеры

        def sort_by_distance(val):
            face = val[1]
            xsum = ysum = zsum = 0.0
            n = len(vertexes[face])

            for vert in vertexes[face]:
                xsum += vert[0]
                ysum += vert[1]
                zsum += vert[2]

            av_x, av_y, av_z = xsum / n, ysum / n, zsum / n
            return av_x * av_x + av_y * av_y + av_z * av_z

        self.color_faces.sort(key=sort_by_distance, reverse=True)

        vertexes = vertexes @ self.render.projection.projection_matrix  # переводим в пространство плоскости отсечения

        vertexes /= vertexes[:, -1].reshape(-1, 1)
        vertexes[(vertexes > 2) | (vertexes < -2)] = 0

        vertexes = vertexes @ self.render.projection.to_screen_matrix

        # def f(val):
        #     face = val[1]
        #     zsum = 0.0
        #
        #     for vert in self.vertexes[face]:
        #         zsum += vert[2]
        #
        #     return zsum / len(self.vertexes[face])
        #
        # self.color_faces.sort(key=f, reverse=True)
        vertexes = vertexes[:, :2]

        # if self.draw_vertexes:
        #     for vertex in vertexes:
        #         if not any_func(vertex, self.render.H_WIDTH, self.render.H_HEIGHT):
        #             pg.draw.circle(self.render.screen, pg.Color('white'), vertex, 2)

        # return vertexes, shaded_colors
        for index, color_face in enumerate(shaded_colors):
            color, face = color_face
            polygon = vertexes[face]
            if not any_func(polygon, self.render.H_WIDTH, self.render.H_HEIGHT):
                pg.draw.polygon(self.render.screen, color, polygon)
                # pg.draw.polygon(self.render.screen, (255, 255, 255), polygon, 1)
                if self.label:
                    text = self.font.render(self.label[index], True, pg.Color('white'))
                    self.render.screen.blit(text, polygon[-1])

    def calculate_shade(self, face, color):
        if not self.shading:
            return color
        verts = self.vertexes[face]
        v1 = Vector3(*verts[0][:3])
        v2 = Vector3(*verts[1][:3])
        v3 = Vector3(*verts[2][:3])
        lite_direction = self.render.light.direction
        normal = Normalize(crossProduct((v2 - v1), (v3 - v1)))

        # if dotProduct(Vector3(*self.vertexes[0][:3]), normal) > 0:
        #     normal *= -1

        val = max(dim, abs(dotProduct(lite_direction, normal)))

        if color[0] * val > 255:
            r = 255
        elif color[0] * val < 0:
            r = 0
        else:
            r = int(color[0] * val)

        if color[1] * val > 255:
            g = 255
        elif color[1] * val < 0:
            g = 0
        else:
            g = int(color[1] * val)

        if color[2] * val > 255:
            b = 255
        elif color[2] * val < 0:
            b = 0
        else:
            b = int(color[2] * val)

        return r, g, b

    def translate(self, pos):
        self.vertexes = self.vertexes @ translate(pos)

    def scale(self, scale_to):
        self.vertexes = self.vertexes @ scale(scale_to)

    def rotate_x(self, angle):
        self.vertexes = self.vertexes @ rotate_x(angle)

    def rotate_y(self, angle):
        self.vertexes = self.vertexes @ rotate_y(angle)

    def rotate_z(self, angle):
        self.vertexes = self.vertexes @ rotate_z(angle)
