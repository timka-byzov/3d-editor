import pygame as pg


class Button:
    def __init__(self, x, y, width, height, message, func, render, new_message=None):
        self.render = render
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.func = func
        self.font = pg.font.SysFont('Arial', 30, bold=True)
        self.message = message
        self.pressed = False
        self.new_message = new_message

    def check_click(self):
        mouse = pg.mouse.get_pos()
        click = pg.mouse.get_pressed()

        if not (click[0] and self.x < mouse[0] < self.x + self.width and self.y < mouse[1] < self.y + self.height):
            self.pressed = False
            return

        if not self.pressed:
            self.func()
            self.pressed = True
            if self.new_message is not None:
                self.change_text()

    def print_text(self):
        text = self.font.render(self.message, True, (0, 0, 0) if self.pressed else (255, 255, 255))
        self.render.screen.blit(text, (self.x, self.y))

    def change_text(self):
        self.new_message, self.message = self.message, self.new_message

    def draw(self):
        #pg.draw.rect(self.render.screen, (255, 255, 255), (self.x, self.y, self.width, self.height), 2)
        self.print_text()
