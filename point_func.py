import math


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __sub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def get_dist(self, other, flag_quad_dist=False):
        quad_dist = (self.x - other.x) ** 2 + (self.y - other.y) ** 2
        if flag_quad_dist:
            return quad_dist
        return math.sqrt(quad_dist)


def check_point_in_face(self, point, face_points):
    pass
