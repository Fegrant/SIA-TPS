# Class color with attributes red, green, blue and methods to convert to and from RGB and CMYK

class Color:
    def __init__(self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue
    
    def rgb(self):
        return [self.red, self.green, self.blue]

    def __str__(self):
        return "({}, {}, {})".format(self.red, self.green, self.blue)