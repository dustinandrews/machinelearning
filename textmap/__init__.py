import random
import numpy as np
from gym import Env
from gym import spaces
from gym.utils import seeding



class Map(Env):
    """
    A 2D map on which objects may be placed
    """    
    
    def __init__(self, height, width):        
        self.visibility = 1  # how far the agent can see
        self.map_init = 2 #0 for obscured, 1 Reserved,  2 for revealed.
        self.height = height
        self.width = width
        self._reset()
        self.observation_space = spaces.Discrete(len(self.data()))
        self.action_space = spaces.Discrete(len(self.actions))        
        self._seed()
        self.metadata = {'render.modes': ['human']}
        self.move_limit = 100
        

    def _close(self):
        return
    
    #def _configure(self):
    #    return
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _reset(self):
        self.map = [["·" for y in range(self.width)] for x in range(self.height)]
        
        self.explored = np.array([[self.map_init for y in range(self.width)] for x in range(self.height)], np.int16)
        symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@0\'&;:~]│─┌┐└┘┼┴┬┤├░▒≡± ⌠≈ · ■'
        #self.symbol_map = {symbols[i]: i/len(symbols) for i in range(len(symbols)) }        
        self.symbol_map = {symbols[i]: i for i in range(len(symbols)) }        
        self.diag_dist = self.get_dist(np.array((0,0), np.float32), np.array((self.height,self.width), np.float32))        
        self.set_spots()
        self.actions = {
                0: {"delta": ( 0, -1), "name": "left"},
                1: {"delta": ( 1, -1), "name": "down-left"},
                2: {"delta": ( 1,  0), "name": "down"},
                3: {"delta": ( 1,  1), "name": "down-right"},
                4: {"delta": ( 0,  1), "name": "right" },
                5: {"delta": ( -1, 1), "name": "up-right"},
                6: {"delta": (-1,  0), "name": "up",},
                7: {"delta": (-1, -1), "name": "up-left"},
                8: {"delta": ( 0,  0), "name": "leave"}
                }
        self.done = False
        self.action_space = {'n': len(self.actions)}
        self.last_action = None
        self.moves = 0
        self.cumulative_score = 0
        self.last_action = "None"
        return self.data()

    #return s_, r, done, info
    def _step(self, n):
        if self.moves > self.move_limit:
            self.done = True
        self.moves += 1
        if n in self.actions:
            # move player
            delta, info = self.actions[n]['delta'], self.actions[n]['name']
            if self.is_in_bounds(self.player + delta):
                self.player += delta
                ex = self.get_indexes_within(self.visibility, self.player)
                self.add_explored(ex)
        
            r = self.score()
                       
            self.explored[self.explored == 1] = 2 # don't double score exploration
            if self.actions[n]["name"] == "leave":
                if np.array_equal( self.player, self.end):
                    r += 100
                    self.done = True
                else:
                    r -= 1
                    
        s_ = self.data()
        self.last_action = self.actions[n]["name"]
        self.cumulative_score += r 
        return s_, r, self.done, info        
        
    def _render(self, mode='human', close=False):
        print("action: {} s: {} t: {}".format(self.last_action,self.cumulative_score, self.moves))
        print("-" * (self.width + 2))
        for i in range(self.height):
            print("|", end="")
            for j in range(self.width):
                if np.all((i,j) == self.player):
                    print("@", end="")
                elif self.explored[i][j] != 0:
                    if  np.all((i,j) == self.end):
                        print("X", end="");
                    else:
                        print(self.map[i][j], end="")
                else:
                    print(" ", end="")
            print("|")
        print("-" * (self.width + 2))
        
    def set_spots(self):
        #self.player = self.getRandomSpot()
        self.player = np.array((1,2), np.float32)
        self.end = self.getRandomSpot()
        while self.player.all() == self.end.all():
            self.end = self.getRandomSpot()
        ex = self.get_indexes_within(self.visibility, self.player)
        self.add_explored(ex)
        #self.setCharacter('@', self.player)
        #self.setCharacter('X', self.end)
        
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

    def angle(self, a, b):
        ang1 = np.arctan2(*a.tolist()[::-1])
        ang2 = np.arctan2(*b.tolist()[::-1])
        return (ang1 - ang2) / (2 * np.pi)
    
    def data_str(self):
        data = []
        for i in range(self.height):            
            for j in range(self.width):
                if np.all((i,j) == self.player):
                    data.append("@")                    
                elif self.explored[i][j] != 0:
                    if  np.all((i,j) == self.end):
                        data.append("X")                        
                    else:                        
                        data.append(self.map[i][j])
                else:
                    data.append(" ")                    
        return data
    
    def data(self, scaled=True):
        scale_factor = 1        
        if scaled:
            scale_factor = 1/len(self.symbol_map)
        data = [self.symbol_map[i] * scale_factor for i in self.data_str()] # visible map
        here = self.map[int(self.player[0])][int(self.player[1])] # spot player is standing.
        data.append(self.symbol_map[here] * scale_factor)
        return np.array(data, dtype=np.float32)
    
    def labels(self):
        labels = [self.get_dist(self.player, self.end)/self.diag_dist,
                self.angle(self.player, self.end)
                ]
        return labels
        #labels.extend(self.player.tolist())
        #labels.extend(self.end.tolist())
    
    def xy_data(self):
        out = np.array((self.player/max(self.height, self.width), self.end/max(self.height, self.width)))
        return out.flatten().tolist()
        
    
    def load_from_data(self,data):
        self.map = [[self.get_symbol(data[x*self.width+y]) for y in range(self.width)] for x in range(self.height)]
        
    def get_symbol(self, in_float):
        return sorted(self.symbol_map, key=lambda s: abs(in_float - self.symbol_map[s]))[0]
    
    def get_indexes_within(self, m_distance, target):
        x = target[0]
        y = target[1]
        ret_list = []
        for i in range (self.height):
            for j in range(self.width):
                dist = abs(j-y) + abs(i-x)
#                print("{},{} -> {}".format(i,j,dist))
                if dist <= m_distance:
                    ret_list.append((i,j))
        return ret_list
    
    def add_explored(self, explored):
        for ex in explored:
            if self.explored[ex[0]][ex[1]] == 0:
                self.explored[ex[0]][ex[1]] = 1
            
    def score(self):
        unique, counts = np.unique(self.explored, return_counts=True)
        d = dict(zip(unique, counts))
        dist_to_goal = self.get_dist(self.player, self.end)
        
        s = (self.diag_dist - dist_to_goal) / 10 # base score per move.
        if 1 in d:
            s = d[1]
        return s
        

    def is_in_bounds(self, xy):
        return xy[0] >= 0 and xy[0] < self.width and xy[1] >= 0 and xy[1] < self.height    
    

if __name__ == '__main__':   
    m = Map(5, 5)
    m.render()
    print(m.step(8))
    print(m.step(0))
    print(m.step(1))
    print(m.step(8))
    m.render()
    
#    for i in m.actions.keys():
#        s_, r, done, info = m.step(i)
#        print("-------", i, r, done, info)
#        m.display()
#    while m.done == False:
#        a = input()
#        if a == "q":
#            break
#        s_, r, done, info = m.step(int(a))
#        print("-------", i, r, done, info)
#        m.display()

      
