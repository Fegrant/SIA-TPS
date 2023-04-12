# Class color with attributes red, green, blue and methods to convert to and from RGB and CMYK

class color:
    def __init__(self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue

    def rgb_to_cmyk(self):
        red = self.red / 255
        green = self.green / 255
        blue = self.blue / 255
        black = 1 - max(red, green, blue)
        if black == 1:
            cyan = 0
            magenta = 0
            yellow = 0
        else:
            cyan = (1 - red - black) / (1 - black)
            magenta = (1 - green - black) / (1 - black)
            yellow = (1 - blue - black) / (1 - black)
        return color(cyan, magenta, yellow, black)

    def cmyk_to_rgb(self):
        cyan = self.cyan
        magenta = self.magenta
        yellow = self.yellow
        black = self.black
        red = 255 * (1 - cyan) * (1 - black)
        green = 255 * (1 - magenta) * (1 - black)
        blue = 255 * (1 - yellow) * (1 - black)
        return color(red, green, blue)
    
    def rgb(self):
        return [self.red, self.green, self.blue]

    def __str__(self):
        return "({}, {}, {})".format(self.red, self.green, self.blue)