import random
import numpy as np
import sys
class Map(object):
    """
    A 2D map on which objects may be placed
    """    
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.map = [["·" for y in range(width)] for x in range(height)]
        symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@0\'&;:~]│─┌┐└┘┼┴┬┤├░▒≡± ⌠≈ · ■'
        self.symbol_map = {symbols[i]: i/len(symbols) for i in range(len(symbols)) }        
        self.diag_dist = self.get_dist(np.array((0,0), np.float32), np.array((height,width), np.float32))        
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

    def get_dist(self, a, b):
        return np.linalg.norm(a - b)
    
    def display(self):
        for i in range(self.height):
            print("".join(self.map[i]))
        sys.stdout.flush()
            
    def angle(self, a, b):
        ang1 = np.arctan2(*a.tolist()[::-1])
        ang2 = np.arctan2(*b.tolist()[::-1])
        return (ang1 - ang2) / (2 * np.pi)
    
    def data(self):
        ords =[self.symbol_map[item] for sublist in self.map for item in sublist]
        #standard = [(o - self.s_mean)/self.s_std for o in ords]        
        return ords
    
    def labels(self):
        labels = [self.get_dist(self.start, self.end)/self.diag_dist,
                self.angle(self.start, self.end)
                ]
        #labels.extend(self.start.tolist())
        #labels.extend(self.end.tolist())
        return labels
    
    def load_from_data(self,data):
        self.map = [[self.get_symbol(data[x*self.width+y]) for y in range(self.width)] for x in range(self.height)]
        
    def get_symbol(self, in_float):
        return sorted(self.symbol_map, key=lambda s: abs(in_float - self.symbol_map[s]))[0]

if __name__ == '__main__':   
    m = Map(5, 5)
    
    mx = m.Map()
    m.display()
    print(m.data())
    print(m.labels())
    
        