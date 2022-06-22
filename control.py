import pygame as pg
import math
from vector import *
from object_3d import Object3D


class Control:
    figure_on_edit = False
    scene_on_rotation = False

    @classmethod
    def update(cls, render, objects):
        mouse = pg.mouse.get_pressed(num_buttons=3)
        mouse_pos = pg.mouse.get_pos()

        if not cls.figure_on_edit:
            SceneControl.control(render.camera, mouse, mouse_pos, objects)
            cls.scene_on_rotation = SceneControl.on_rotation
        if not cls.scene_on_rotation:
            FiguresControl.control(mouse, mouse_pos, objects)
            cls.figure_on_edit = FiguresControl.mouse_hold


class SceneControl:
    prev_x = None
    prev_y = None
    on_rotation = False

    # @classmethod
    # def update(cls, mouse, mouse_pos, objects):
    #     cls.control(mouse, mouse_pos, objects)

    @classmethod
    def control(cls, camera, mouse, mouse_pos, objects: list):
        if mouse[0]:
            if cls.prev_x is None:
                cls.prev_x, cls.prev_y = mouse_pos
            else:
                x, y = pg.mouse.get_pos()
                kx = -(x - cls.prev_x) / 35
                ky = -(y - cls.prev_y) / 35

                camera.camera_rotate_scene_y(kx * math.pi / 15)
                camera.camera_rotate_scene_x(ky * math.pi / 15)

                cls.prev_x = x
                cls.prev_y = y

                cls.on_rotation = True

        else:
            cls.on_rotation = False
            cls.prev_x = None
            cls.prev_y = None


class FiguresControl:
    mouse_hold = False
    editing_object = None

    @classmethod
    def control(cls, mouse, mouse_pos, objects):
        # cls.control_scale(mouse, mouse_pos, objects)

        if not mouse[0] and not mouse[2]:
            if cls.mouse_hold:
                cls.mouse_hold = False
                cls.editing_object.on_scale = False
                cls.editing_object.on_rotation = False
                cls.editing_object.on_translate = False
            return

        # for object in objects:
        #     object.mouse_scale()

        x, y = mouse_pos
        clicked_obj, is_point_click = Object3D.check_click(Vector2(x, y), objects)
        if clicked_obj is not None:
            if not cls.mouse_hold:  # new_move
                cls.mouse_hold = True
                # sbros
                cls.reset_objects(objects)

                clicked_obj.is_highlighted = True
                cls.object_on_edit = True
                cls.editing_object = clicked_obj

                # new_move
                if is_point_click and mouse[0]:
                    clicked_obj.on_scale = True

                elif mouse[0]:
                    clicked_obj.on_translate = True

                elif mouse[2]:
                    clicked_obj.on_rotation = True

        else:
            if not cls.mouse_hold:
                cls.reset_objects(objects)

    @classmethod
    def reset_objects(cls, objects):
        for obj in objects:
            obj.on_scale = False
            obj.on_rotation = False
            obj.is_highlighted = False
            obj.prev_x = None
            obj.prev_y = None
    #
    # @classmethod
    # def reset_highlights(cls, objects):
    #     for obj in objects:
    #         obj.is_highlighted = False

    # @classmethod
    # hi

    # @classmethod
    # def control_scale(cls, mouse, mouse_pos, objects):
    #
