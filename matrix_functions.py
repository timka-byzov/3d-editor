import math
import numpy as np
from vector import Vector3

def translate(pos):
    tx, ty, tz = pos
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1]
    ])


def rotate_x(a):
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(a), math.sin(a), 0],
        [0, -math.sin(a), math.cos(a), 0],
        [0, 0, 0, 1]
    ])


def rotate_y(a):
    return np.array([
        [math.cos(a), 0, -math.sin(a), 0],
        [0, 1, 0, 0],
        [math.sin(a), 0, math.cos(a), 0],
        [0, 0, 0, 1]
    ])


def rotate_z(a):
    return np.array([
        [math.cos(a), math.sin(a), 0, 0],
        [-math.sin(a), math.cos(a), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def scale(n):
    return np.array([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, n, 0],
        [0, 0, 0, 1]
    ])


def rotate_around_vector(v: Vector3, angle):
    cos = math.cos(angle)
    sin = math.sin(angle)

    return np.array([
        [cos + (1 - cos) * v.x * v.x, (1 - cos) * v.x * v.y - sin * v.z, (1 - cos) * v.x * v.z + sin * v.y, 0],
        [(1 - cos) * v.y * v.x + sin * v.z, cos + (1 - cos) * v.y * v.y, (1 - cos) * v.y * v.z - sin * v.x, 0],
        [(1 - cos) * v.z * v.x - sin * v.y, (1 - cos) * v.z * v.y + sin * v.x, cos + (1 - cos) * v.z * v.z, 0],
        [0, 0, 0, 1]
    ])