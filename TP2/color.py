# Class color with attributes red, green, blue and methods to convert to and from RGB and CMYK

class Color:
    def __init__(self, red, green, blue):
        self.rgb = [red, green, blue]
    
    def rgb(self):
        return self.rgb

    def __str__(self):
        return "({}, {}, {})".format(self.red, self.green, self.blue)