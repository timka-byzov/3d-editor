import pygame as pg
from matrix_functions import *
from numba import njit
from vector import *

dim = 0.01
EPS = 1e-3


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

    @classmethod
    def check_point_in_face_on_plane(cls, point_coords: Vector2, face_coords):  # (edges(x, y))
        middle_of_edge = (Vector2(*face_coords[0]) + Vector2(*face_coords[1])) * 0.5

        point = Vector2(*point_coords)
        vec = middle_of_edge - point

        intersections_count = 0
        for i in range(len(face_coords) - 1):
            p1 = Vector2(*face_coords[i])
            p2 = Vector2(*face_coords[i + 1])

            edge_vec = p2 - p1

            a = np.array([[vec.x, -edge_vec.x], [vec.y, -edge_vec.y]])
            b = np.array([p1.x - point.x, p1.y - point.y])
            res = np.linalg.solve(a, b)
            t, q = res[0], res[1]

            if t > 0 and 0 < q < 1:
                intersections_count += 1
        return intersections_count % 2 == 1


    @classmethod
    def check_two_edges_intersection(cls, vertexes1, vertexes2):
        for i in range(len(vertexes1)):
            vec1 = Vector2(*vertexes1[(i + 1) % len(vertexes1)]) - Vector2(vertexes1[i])
            for j in range(len(vertexes2)):
                vec2 = Vector2(*vertexes2[])



    @classmethod
    def draw_objects(cls, render, objects: list):

        vertexes = []
        objects_colors_faces = []

        for i, object in enumerate(objects):
            object.movement()

            shaded_colors = [(object.calculate_shade(face_color[1], face_color[0]), face_color[1]) for face_color in
                             object.color_faces]

            obj_verts = object.vertexes @ render.camera.camera_matrix()  # переводим в пространство камеры
            vertexes.append(obj_verts)
            objects_colors_faces += [(i, shaded_color) for shaded_color in shaded_colors]

        # def sort_by_distance(color_faces):
        #     object_num, color_face = color_faces
        #     face = color_face[1]
        #
        #     xsum = ysum = zsum = 0.0
        #     n = len(vertexes[object_num][face])
        #
        #     for vert in vertexes[object_num][face]:
        #         xsum += vert[0]
        #         ysum += vert[1]
        #         zsum += vert[2]
        #
        #     av_x, av_y, av_z = xsum / n, ysum / n, zsum / n
        #
        #     return av_x * av_x + av_y * av_y + av_z * av_z
        #
        # objects_colors_faces.sort(key=sort_by_distance, reverse=True)

        for i in range(len(objects_colors_faces) - 1):

            for j in range(i + 1, len(objects_colors_faces)):

                object1, tup1 = objects_colors_faces[i]
                color1, vertexes_set1 = tup1
                vertexes1 = [vertexes[object1][i] for i in vertexes_set1]

                object2, tup2 = objects_colors_faces[j]
                color2, vertexes_set2 = tup2
                vertexes2 = [vertexes[object2][i] for i in vertexes_set2]

                t1 = not cls.check_face_camera_same_side_in_camera_space(vertexes1, vertexes2)
                t2 = cls.check_face_camera_same_side_in_camera_space(vertexes2, vertexes1)
                t3 = cls.check_intersection(cls.project_on_plain(render, vertexes1),
                                            cls.project_on_plain(render, vertexes2))
                # t4 = cls.find_point_inside(cls.project_on_plain(render, vertexes2),
                #                            cls.project_on_plain(render, vertexes1))

                if t1 and t2:
                    objects_colors_faces[i], objects_colors_faces[j] = objects_colors_faces[j], objects_colors_faces[i]

        objects_colors_faces.reverse()

        for vert_set_num in range(len(vertexes)):
            temp_verts = vertexes[vert_set_num]
            temp_verts = temp_verts @ render.projection.projection_matrix
            temp_verts /= temp_verts[:, -1].reshape(-1, 1)
            temp_verts[(temp_verts > 2) | (temp_verts < -2)] = 0
            temp_verts = temp_verts @ render.projection.to_screen_matrix
            temp_verts = temp_verts[:, :2]
            vertexes[vert_set_num] = temp_verts

        for index, color_face in enumerate(objects_colors_faces):
            object_num, tup = color_face
            color, face = tup

            # font = pg.font.SysFont('Comic Sans MS', 20)
            # for i in range(len(face)):
            #     vertex = vertexes[object_num][i]
            #     vert = face[i]
            #     render.screen.blit(font.render(str(vert), False, (255, 255, 255)), (vertex[0], vertex[1]))

            polygon = vertexes[object_num][face]
            if not any_func(polygon, render.H_WIDTH, render.H_HEIGHT):
                pg.draw.polygon(render.screen, color, polygon)
                # pg.draw.polygon(self.render.screen, (255, 255, 255), polygon, 1)
                # if self.label:
                #     text = self.font.render(self.label[index], True, pg.Color('white'))
                #     self.render.screen.blit(text, polygon[-1])

    @classmethod
    def project_on_plain(cls, render, vertexes):
        vertexes = vertexes @ render.projection.projection_matrix
        vertexes /= vertexes[:, -1].reshape(-1, 1)
        vertexes[(vertexes > 2) | (vertexes < -2)] = 0
        vertexes = vertexes @ render.projection.to_screen_matrix
        return vertexes[:, :2]

    @classmethod
    def get_face_equation(cls, vertexes):
        p1 = Vector3(*vertexes[1][:3])
        p2 = Vector3(*vertexes[0][:3])
        p3 = Vector3(*vertexes[2][:3])

        v1 = p2 - p1
        v2 = p3 - p1

        t1 = -np.linalg.det(np.array([[v1.y, v1.z], [v2.y, v2.z]]))
        t2 = -np.linalg.det(np.array([[v1.x, v1.z], [v2.x, v2.z]]))
        t3 = -np.linalg.det(np.array([[v1.x, v1.y], [v2.x, v2.y]]))

        return [t1, -t2, t3, -p1.x * t1 + p1.y * t2 - p1.z * t3]

    @classmethod
    def check_face_camera_same_side_in_camera_space(cls, P_vertexes, Q_vertexes):
        Q_equation = cls.get_face_equation(Q_vertexes)

        def equation(face_equation, point: Vector3):
            return face_equation[0] * point.x + face_equation[1] * point.y + face_equation[2] * point.z + face_equation[
                3]

        camera_side = equation(Q_equation, Vector3(0, 0, 0))

        plus_count, minus_count, nul_count = 0, 0, 0
        for i in range(len(P_vertexes)):
            res = equation(Q_equation, Vector3(*P_vertexes[i][:3]))
            if abs(res) < EPS:
                nul_count += 1
                continue

            if res < 0:
                minus_count += 1
            else:
                plus_count += 1

        return (minus_count and camera_side < 0) or (plus_count and camera_side > 0)

    @classmethod
    def check_intersection(cls, vertexes1, vertexes2):
        for j in range(len(vertexes2)):
            cls.find_point_inside(vertexes1, Vector2(*vertexes2[j]))

    @classmethod
    def find_point_inside(cls, vertexes, point: Vector2):
        def equation(p1: Vector2, p2: Vector2, point: Vector2):
            return (point.x - p1.x) * (p2.y - p1.y) - (point.y - p1.y) * (p2.x - p1.x)

        minus_count = 0
        plus_count = 0
        nul_count = 0

        for i in range(len(vertexes)):
            p1 = Vector2(*vertexes[i])
            p2 = Vector2(*vertexes[(i + 1) % len(vertexes)])

            res = equation(p1, p2, point)
            if abs(res) < EPS:
                nul_count += 1
                continue

            if res > 0:
                plus_count += 1
            else:
                minus_count += 1

            # if plus_count + nul_count == len(vertexes2) or \
            #         minus_count + nul_count == len(vertexes2):
            #     return False

        return True

    def movement(self):
        if self.movement_flag:
            self.rotate_y(-(pg.time.get_ticks() % 0.005))
        pass

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
