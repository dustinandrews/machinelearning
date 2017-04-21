import random
import numpy as np
class Map(object):
    """
    A 2D map on which objects may be placed
    """
    
    
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.map = [["·" for y in range(width)] for x in range(height)]
        symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@0\'&;:~]│─┌┐└┘┼┴┬┤├░▒≡± ⌠≈ · ■'
        self.symbolmap = {symbols[i]: i/len(symbols) for i in range(len(symbols)) }
        self.set_spots()
        
    def set_spots(self):
        self.start = self.getRandomSpot()
        self.end = self.getRandomSpot()
        while self.start.all() == self.end.all():
            self.end = self.getRandomSpot()
        self.setCharacter('@', self.start)
        self.setCharacter('X', self.end)
        
    def Map(self):
        return self.map
    
    def getRandomSpot(self):
        x = random.randint(0, self.height - 1)
        y = random.randint(0, self.width - 1)
        return np.array((x,y), dtype=np.int32)
    
    def setCharacter(self,c,coord):
        self.map[coord[0]][coord[1]] = c

    def getDist(self, a, b):
        return np.linalg.norm(a - b)
    
    def display(self):
        for i in range(self.height):
            print("".join(self.map[i]))
            
    def angle(self, a, b):
        ang1 = np.arctan2(*a.tolist()[::-1])
        ang2 = np.arctan2(*b.tolist()[::-1])
        return (ang1 - ang2) % (2 * np.pi)
    
    def data(self):
        ords =[self.symbolmap[item] for sublist in self.map for item in sublist]
        #standard = [(o - self.s_mean)/self.s_std for o in ords]        
        return ords
    
    def labels(self):
        labels = [self.getDist(self.start, self.end),
                self.angle(self.start, self.end)
                ]
        labels.extend(self.start.tolist())
        labels.extend(self.end.tolist())
        return labels

if __name__ == '__main__':   
    m = Map(15, 10)
    
    mx = m.Map()
    m.display()
    print(len(m.data()))
    print(len(m.labels()))