import pygame as pg
from matrix_functions import *
from numba import njit
from vector import *
import numpy as np
import math

dim = 0.01
EPS = 1e-4


@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))


class Object3D:
    vertexes = []
    projected_vertexes = []
    objects_colors_faces = []
    render = None

    def __init__(self, render, shading=True, vertexes='', faces=''):
        self.render = render
        self.vertexes = np.array([np.array(v) for v in vertexes])
        self.faces = np.array([np.array(face) for face in faces])
        self.shading = shading
        self.is_highlighted = False
        self.font = pg.font.SysFont('Arial', 30, bold=True)
        self.color_faces = [(pg.Color('orange'), face) for face in self.faces]
        self.movement_flag, self.draw_vertexes = False, False
        self.label = ''
        self.on_scale = False
        self.on_rotation = False
        self.prev_x = None
        self.prev_y = None
        self.scale_value = 1
        self.transform_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def get_vertexes(self):
        return self.vertexes @ self.transform_matrix

    # def add_face(self, point1, point2, old_points_nums):
    #     last_num = len(self.vertexes)
    #     self.vertexes.append([*point1, 1])
    #     self.vertexes.append([*point2, 1])
    #     self.faces += [last_num] + old_points_nums + [last_num + 1]

    @classmethod
    def make_faces_intersections_in_space(cls):
        def get_point_in_intersection(equation1, equation2):
            for i in range(3):
                for j in range(3):
                    if i == j:
                        continue
                    a = np.array([[equation1[i], equation1[j]], [equation2[i], equation2[j]]])
                    if np.linalg.det(a) == 0:
                        continue
                    b = np.array([-equation1[3], -equation2[3]])

                    solution = np.linalg.solve(a, b)
                    res = [0] * 3
                    res[i], res[j] = solution[0], solution[1]

                    return res

        res_faces = []
        res_vertexes = cls.vertexes[:]
        for i in range(len(cls.objects_colors_faces)):
            object1, tup1 = cls.objects_colors_faces[i]
            color1, vertexes_set1 = tup1
            vertexes1 = [cls.vertexes[object1][_] for _ in vertexes_set1]
            equation1 = cls.get_face_equation(vertexes1)

            res_faces.append((object1, (color1, vertexes_set1)))

            for j in range(len(cls.objects_colors_faces)):

                if i == j:
                    continue

                object2, tup2 = cls.objects_colors_faces[j]
                color2, vertexes_set2 = tup2
                vertexes2 = [cls.vertexes[object2][_] for _ in vertexes_set2]
                equation2 = cls.get_face_equation(vertexes2)

                norm_v1 = Vector3(equation1[0], equation1[1], equation1[2])
                norm_v2 = Vector3(equation2[0], equation2[1], equation2[2])

                if norm_v1 * norm_v2 == 0:
                    continue

                new_points = []
                left_points = []
                right_points = []
                is_left_points = True
                dir_vec = crossProduct(norm_v1, norm_v2)
                point = Vector3(*get_point_in_intersection(equation1, equation2))

                for k in range(len(vertexes2)):
                    p1 = Vector3(*vertexes2[k][:3])
                    p2 = Vector3(*vertexes2[(k + 1) % len(vertexes2)][:3])

                    edge_vec = p2 - p1

                    a = np.array([[dir_vec.x, -edge_vec.x], [dir_vec.y, -edge_vec.y]])
                    b = np.array([p1.x - point.x, p1.y - point.y])

                    if is_left_points:
                        left_points.append(vertexes2[k])
                    else:
                        right_points.append(vertexes2[k])

                    try:
                        t, q = np.linalg.solve(a, b)
                        if -EPS < q < 1 + EPS:
                            t = p1 + edge_vec * q
                            coords = cls.project_on_plain(cls.render, np.array([[t.x, t.y, t.z, 1]]))
                            pg.draw.circle(cls.render.screen, (255, 255, 255), (coords[0][0], coords[0][1]), 5)
                            new_points.append(p1 + edge_vec * q)
                            is_left_points = not is_left_points

                    except:
                        # t = p1 + edge_vec * q
                        # coords = cls.project_on_plain(cls.render, np.array([[t.x, t.y, t.z, 1]]))
                        # pg.draw.circle(cls.render.screen, (255, 255, 255), (coords[0][0], coords[0][1]), 5)
                        continue

                if len(new_points) < 2:
                    res_faces.append((object2, (color2, vertexes_set2)))
                    continue
                cls.add_new_fictive_face(res_vertexes, new_points[0], new_points[1], left_points, color2, object2)
                cls.add_new_fictive_face(res_vertexes, new_points[0], new_points[1], right_points, color2, object2)

        cls.objects_colors_faces = res_faces
        cls.vertexes = res_vertexes

    @classmethod
    def add_new_fictive_face(cls, vertexes, point1, point2, old_vertexes, color, obj):
        obj_count = len(cls.objects_colors_faces)
        vert_set = list(range(len(old_vertexes) + 2))
        obj_verts = [[point1.x, point1.y, point1.z, 1]] + old_vertexes + [[point2.x, point2.y, point2.z, 1]]
        vertexes.append(obj_verts)
        cls.objects_colors_faces.append((obj_count, (color, np.array(vert_set))))

    @classmethod
    def make_svalka(cls, render, objects):
        cls.vertexes = []
        cls.projected_vertexes = []
        cls.objects_colors_faces = []

        for i, object in enumerate(objects):
            object.movement()
            shaded_colors = [(object.calculate_shade(face_color[1], face_color[0]), face_color[1]) for face_color in
                             object.color_faces]
            obj_verts = object.get_vertexes() @ render.camera.camera_matrix()  # переводим в пространство камеры
            cls.projected_vertexes.append(cls.project_on_plain(render, obj_verts))
            cls.vertexes.append(obj_verts)
            cls.objects_colors_faces += [(i, shaded_color) for shaded_color in shaded_colors]

    def self_update(self):
        self.mouse_scale()
        self.mouse_rotate()

    @classmethod
    def update(cls, render, objects: list):

        # cls.control(objects_colors_faces, projected_vertexes, objects)
        for object in objects:
            object.self_update()
        cls.render = render
        cls.make_svalka(render, objects)
        #cls.make_faces_intersections_in_space()
        cls.sort_faces()
        cls.draw_objects(render, objects)

    @classmethod
    def check_point_in_face_on_plane(cls, point: Vector2, face_coords):  # (edges(x, y))
        middle_of_edge = Vector2(*face_coords[0]) + (Vector2(*face_coords[1]) - Vector2(*face_coords[0])) * 0.5

        # point = Vector2(*point_coords)
        vec = middle_of_edge - point

        intersections_count = 0
        for i in range(len(face_coords)):
            p1 = Vector2(*face_coords[i])
            p2 = Vector2(*face_coords[(i + 1) % len(face_coords)])

            edge_vec = p2 - p1

            a = np.array([[vec.x, -edge_vec.x], [vec.y, -edge_vec.y]])
            b = np.array([p1.x - point.x, p1.y - point.y])

            try:
                t, q = np.linalg.solve(a, b)
                if t > -EPS and -EPS < q < 1 + EPS:
                    intersections_count += 1
            except:
                continue

        return intersections_count % 2 == 1

    @classmethod
    def check_points_intersection(cls, vertexes1, vertexes2):
        for i in range(len(vertexes1)):
            point = Vector2(*vertexes1[i])
            if cls.check_point_in_face_on_plane(point, vertexes2):
                return True

        for j in range(len(vertexes2)):
            point = Vector2(*vertexes2[j])
            if cls.check_point_in_face_on_plane(point, vertexes1):
                return True

        return False

    @classmethod
    def check_edges_intersection(cls, vertexes1, vertexes2):
        for i in range(len(vertexes1)):
            point1 = Vector2(*vertexes1[i])
            vec1 = Vector2(*vertexes1[(i + 1) % len(vertexes1)]) - Vector2(*vertexes1[i])
            for j in range(len(vertexes2)):
                point2 = Vector2(*vertexes2[j])
                vec2 = Vector2(*vertexes2[(j + 1) % (len(vertexes2))]) - Vector2(*vertexes2[j])

                a = np.array([[vec1.x, -vec2.x], [vec1.y, -vec2.y]])
                b = np.array([point2.x - point1.x, point2.y - point1.y])

                try:
                    t, q = np.linalg.solve(a, b)

                    if -EPS < t < 1 + EPS and -EPS < q < 1 + EPS:
                        return True
                except:
                    continue
        return False

    @classmethod
    def sort_faces(cls):
        for k in range(len(cls.objects_colors_faces) - 1):
            for i in range(k, len(cls.objects_colors_faces)):

                object1, tup1 = cls.objects_colors_faces[i]
                color1, vertexes_set1 = tup1
                vertexes1 = [cls.vertexes[object1][_] for _ in vertexes_set1]

                for j in range(k, len(cls.objects_colors_faces)):
                    if i == j:
                        continue

                    object2, tup2 = cls.objects_colors_faces[j]
                    color2, vertexes_set2 = tup2
                    vertexes2 = [cls.vertexes[object2][_] for _ in vertexes_set2]

                    t1 = cls.check_face_camera_same_side_in_camera_space(vertexes1, vertexes2)
                    t2 = not cls.check_face_camera_same_side_in_camera_space(vertexes2, vertexes1)
                    t3 = cls.check_intersection([cls.projected_vertexes[object1][_] for _ in vertexes_set1],
                                                [cls.projected_vertexes[object2][_] for _ in vertexes_set2])

                    if t3 and t2 and t1:
                        break

                else:
                    cls.objects_colors_faces[k], cls.objects_colors_faces[i] = cls.objects_colors_faces[i], \
                                                                               cls.objects_colors_faces[k]
                    break

    @classmethod
    def draw_objects(cls, render, objects):
        print(cls.objects_colors_faces)

        for index, color_face in enumerate(cls.objects_colors_faces):
            object_num, tup = color_face
            color, face = tup

            # font = pg.font.SysFont('Comic Sans MS', 20)
            # for i in range(len(face)):
            #     vertex = vertexes[object_num][i]
            #     vert = face[i]
            #     render.screen.blit(font.render(str(vert), False, (255, 255, 255)), (vertex[0], vertex[1]))

            polygon = cls.projected_vertexes[object_num][face]
            if not any_func(polygon, render.H_WIDTH, render.H_HEIGHT):
                pg.draw.polygon(render.screen, color, polygon)
                if objects[object_num].is_highlighted:
                    pg.draw.polygon(render.screen, (255, 255, 255), polygon, 2)
                    for vertex in polygon:
                        pg.draw.circle(render.screen, (255, 255, 255), vertex, 5)
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

        return (minus_count > 0 and camera_side <= 0) or (plus_count > 0 and camera_side >= 0)

    @classmethod
    def check_intersection(cls, vertexes1, vertexes2):
        t1 = cls.check_edges_intersection(vertexes1, vertexes2)
        t2 = cls.check_points_intersection(vertexes1, vertexes2)
        return t1 or t2

    def movement(self):
        if self.movement_flag:
            self.rotate_y(-(pg.time.get_ticks() % 0.005) * 10)
            # self.rotate_x(-(pg.time.get_ticks() % 0.005) * 5)

        pass

    def calculate_shade(self, face, color):
        if not self.shading:
            return color
        verts = self.vertexes[face]
        v1 = Vector3(*verts[0][:3])
        v2 = Vector3(*verts[1][:3])
        v3 = Vector3(*verts[2][:3])
        lite_direction = Vector3(*self.render.camera.forward[:3])
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
        self.transform_matrix = self.transform_matrix @ translate(pos)

    def scale(self, scale_to):
        self.vertexes = self.vertexes @ scale(scale_to)

    def rotate_x(self, angle):
        self.vertexes = self.vertexes @ rotate_x(angle)

    def rotate_y(self, angle):
        self.vertexes = self.vertexes @ rotate_y(angle)

    def rotate_z(self, angle):
        self.vertexes = self.vertexes @ rotate_z(angle)

    def rotate_global_x(self, angle):
        self.transform_matrix = self.transform_matrix @ rotate_x(angle)

    def rotate_global_y(self, angle):
        self.transform_matrix = self.transform_matrix @ rotate_y(angle)

    def rotate_global_z(self, angle):
        self.transform_matrix = self.transform_matrix @ rotate_z(angle)

    # @classmethod
    # def find_visible_points(cls, objects_colors_faces, projected_vertexes, render):
    #     for i in range(len(objects_colors_faces) - 1, -1, -1):
    #         object, tup = objects_colors_faces[i]
    #         color, vertexes_set = tup
    #         face = [projected_vertexes[object][_] for _ in vertexes_set]
    #
    #         count = 0
    #         for vertex in projected_vertexes:
    #             if cls.check_point_in_face_on_plane(Vector2(vertex[0], vertex[1]), face):
    #                 count += 1
    #

    @classmethod
    def check_point_click_inside(cls, point: Vector2, face, radius):
        for vertex in face:
            point2 = Vector2(vertex[0], vertex[1])
            d = point2.get_dist(point)
            if d <= radius:
                return True
        return False

    @classmethod
    def check_point_click_outside(cls, point: Vector2, object_vertexes, radius):
        for vertex in object_vertexes:
            point2 = Vector2(vertex[0], vertex[1])
            if point2.get_dist(point) <= radius:
                return True

        return False

    @classmethod
    def check_click(cls, point: Vector2, objects):
        for i in range(len(cls.objects_colors_faces) - 1, -1, -1):
            object, tup = cls.objects_colors_faces[i]
            color, vertexes_set = tup
            face = [cls.projected_vertexes[object][_] for _ in vertexes_set]

            # for i in range(len(objects)):
            #     objects[i].is_highlighted = False

            if cls.check_point_in_face_on_plane(point, face):
                # return object
                # if not objects[object].is_highlighted:
                #     for j in range(len(objects)):
                #         objects[j].is_highlighted = False
                #
                #     objects[object].is_highlighted = True
                #     print(object)

                if cls.check_point_click_inside(point, face, 15):
                    return objects[object], True
                return objects[object], False

            if cls.check_point_click_outside(point, cls.projected_vertexes[object], 15):
                return objects[object], True

        return None, False

    # @classmethod
    # def
    #

    # --contol --
    def mouse_scale(self):
        def clip(scale_value):
            if scale_value < 0.7:
                return 0.7
            if scale_value > 2:
                return 2
            return scale_value

        if self.on_scale:
            if self.prev_x is None:
                self.prev_x, self.prev_y = pg.mouse.get_pos()
            else:
                x, y = pg.mouse.get_pos()
                scale_delta = 1 + (-x + self.prev_x) / 100
                new_scale_value = clip(self.scale_value * scale_delta)
                # self.scale_value =
                self.scale(self.scale_value / new_scale_value)
                self.scale_value = new_scale_value
                self.prev_x, self.prev_y = x, y

    def mouse_rotate(self):
        if self.on_rotation:
            if self.prev_x is None:
                self.prev_x, self.prev_y = pg.mouse.get_pos()
            else:
                x, y = pg.mouse.get_pos()
                dx = -(x - self.prev_x)
                dy = -(y - self.prev_y)
                if abs(dx) > abs(dy):
                    self.rotate_y(math.pi / 180 * dx)
                else:
                    self.rotate_x(math.pi / 180 * dy)
                self.prev_x, self.prev_y = x, y
