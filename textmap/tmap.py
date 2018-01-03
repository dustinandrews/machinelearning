import numpy as np
np.set_printoptions(threshold=np.nan)
from gym import Env
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt



class Map(Env):
    """
    A 2D map on which objects may be placed
    """
    done = False
    visibility = 1
    map_init = 2 #0 for obscured, 1 Reserved,  2 for revealed.
    curriculum=None
    cost_of_living = 0.1


    def __init__(self, height, width):
        self.width = width
        self.height = height
        self._reset()
        self.observation_space = spaces.Discrete(len(self.data()))
        self.action_space = spaces.Discrete(len(self._actions))
        self._seed()
        self.metadata = {'render.modes': ['human', 'graphic']}
        self.move_limit = height + width

    def __del__(self):
        # don't need the base class to do anything fancy.
        pass

    def _close(self):
        return

    #def _configure(self):
    #    return

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        """Reset the simulation

        Start a fresh random map


        Args:
            curriculum (int): An int indicating the stage of training or None.
                If present will indicate how many steps from the goal the
                player will be placed.

        Returns:
            numpy.array: The state of the game
        """
        self.explored = np.ones([self.height,self.width]) * self.map_init
        #symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@0\'&;:~]│─┌┐└┘┼┴┬┤├░▒≡± ⌠≈ · ■'
        self.symbols = '.x@'
        #self.symbol_map = {symbols[i]: i/len(symbols) for i in range(len(symbols)) }
        self._num_categories = len(self.symbols)
        self.symbol_map = {self.symbols[i]: i for i in range(len(self.symbols)) }
        self.diag_dist = self.get_dist(np.array((0,0), np.float32), np.array((self.width,self.height), np.float32))
        self.set_spots()
        self._actions = {
                # Maps to numpad
                4: {"delta": ( 0, -1), "name": "left"},
                1: {"delta": ( 1, -1), "name": "down-left"},
                2: {"delta": ( 1,  0), "name": "down"},
                3: {"delta": ( 1,  1), "name": "down-right"},
                6: {"delta": ( 0,  1), "name": "right" },
                9: {"delta": ( -1, 1), "name": "up-right"},
                8: {"delta": (-1,  0), "name": "up",},
                7: {"delta": (-1, -1), "name": "up-left"},
                #5: {"delta": ( 0,  0), "name": "leave"},
                }
        self.action_index = list(self._actions.keys())
        self.action_index.sort()
        self.done = False
        self.action_space = {'n': len(self._actions)}
        self.last_action = None
        self.moves = 0
        self.last_score = 0
        self.cumulative_score = 0
        self.last_action = {'name': 'None', 'delta': (0,0)}
        self.found_exit = False
        self.cumulative_score = 0
        return self.data()

    #return s_, r, done, info
    def _step(self, a: int):
        if self.done:
            raise RuntimeError('Simulation is ended. Must call reset.')
        n = self.action_index[a]
        if self.moves > self.move_limit:
            self.cumulative_score -= 1
            self.done = True
        self.moves += 1
        old_player = np.copy(self.player)
        r = -0.01
#        print(type(self._actions))
#        print(self._actions.keys())
#        print(list(self._actions.keys()))
        if n in self._actions:
            # move player
            delta, info = self._actions[n]['delta'], self._actions[n]['name']
            if self.is_in_bounds(self.player + delta):
                self.player += delta
                ex = self.get_indexes_within(self.visibility, self.player)
                self.add_explored(ex)
                r = self.score(old_player)
            else:
                r = -1 #penalty for bumping wall
                self.done = True

            self.explored[self.explored == 1] = 2 # don't double score exploration
            #if self._actions[n]["name"] == "leave":

        s_ = self.data()
        self.last_action = self._actions[n]
        self.cumulative_score += r
       # self.last_render = self.get_render_string()
        return s_, r, self.done, info

    def score(self, last_pos):
        r = -self.cost_of_living
        if not self.found_exit:
            if np.array_equal(self.player, self.end):
                r = 1
                self.found_exit = True
                self.done = True
        self.last_score = r
        return r

    def _render(self, mode='human', close=False):
        if mode == 'human':
            print(self.get_render_string())
        else:
            out, ann = self.data_collapsed()
            plt.imshow(out, interpolation='nearest') #cmap='magma'
            for a in ann:
                plt.annotate(a[0], xy=a[1], ha='center', va='center')
                plt.axis('off')


    def get_render_string(self):
        render_string = ""
        render_string += ("action: {} s: {}/{} t: {} done: {}\n".format(self.last_action["name"], self.last_score, self.cumulative_score, self.moves, self.done))
        render_string += ("-" * (self.height + 2))
        render_string += ("\n")

        d = self.data()
        out_data = np.zeros((self.height, self.width))
        for layer_num in range(self._num_categories):
            layer = d[:,:,layer_num]
            out_data[layer > 0] = layer_num

        for j in range(self.height):
            render_string += '|'
            for i in range(self.width):
                index = int(out_data[j,i])
                render_string += self.symbols[index]
            render_string += '|\n'

        render_string += "-" * (self.height + 2)
        self.last_render = render_string
        return render_string


    def set_spots_new(self):
        choices = np.random.choice(
                np.arange(np.product(self.height, self.width)),
                size=len(self.symbols))

    def set_spots(self):
        self.end = self.get_random_spot()

        if not self.curriculum:
            self.curriculum = np.max([self.width, self.height])

        self.player = self.get_spot_near(self.end, self.curriculum)

        ex = self.get_indexes_within(self.visibility, self.player)
        self.add_explored(ex)
        #self.set_character('@', self.player)
        #self.set_character('X', self.end)


    def get_spot_near(self, origin, distance: int):
        pos = origin.copy()
        safety_valve = 0
        while np.array_equal(pos, origin):
            x = origin[0]
            y = origin[1]
            direction = [-1,1]

            x_dist = np.random.randint(1,distance+1)
            x_dir = direction[np.random.randint(2)]

            y_dist = np.random.randint(1,distance+1)
            y_dir = direction[np.random.randint(2)]

            x_vector = x_dist * x_dir
            y_vector = y_dist * y_dir

            new_x = x + x_vector
            new_y = y + y_vector

            new_x = np.min([new_x, self.height - 1])
            new_x = np.max([0, new_x])

            new_y = np.min([new_y, self.width -1])
            new_y = np.max([0, new_y])

            pos = np.array([new_x, new_y])
            if safety_valve > 10:
                if safety_valve > 11:
                    raise ValueError('Unexpected bug in module, can not find valid empty spot.')
                valid_choices = self.get_indexes_within(distance, origin)
                pos = np.array(valid_choices[np.random.choice(len(valid_choices))])
            safety_valve += 1
        return pos

    def get_random_spot(self):
        x = np.random.randint(self.height)
        y = np.random.randint(self.width)
        return np.array((x,y), dtype=np.int32)

    def get_dist(self, a, b):
        # Manhattan dist
        m = np.sum(np.abs(a-b))
        return m

        # Linear dist
        #return np.linalg.norm(a - b)

    def angle(self, a, b):
        ang1 = np.arctan2(*a.tolist()[::-1])
        ang2 = np.arctan2(*b.tolist()[::-1])
        return (ang1 - ang2) / (2 * np.pi)


    def data(self):
        #return self.data2d()
        return self.data_n_dim()

    def data2d(self):
        data = np.zeros((self.width, self.height), dtype=np.int32)
        for i in range(self.width):
            for j in range(self.height):
                if np.all((i,j) == self.player):
                    data[i,j] = self.symbol_map['@']
                elif self.explored[i][j] != 0:
                    if  np.all((i,j) == self.end):
                        data[i,j] = self.symbol_map['X']
                    else:
                        data[i,j] = self.symbol_map['·']
                else:
                    data[i,j] = self.symbol_map[' ']
        data = np.array(data, dtype=np.int32)
        return data

    def data_n_dim(self):
        shape = (self.height, self.width, self._num_categories)
        data = np.zeros(shape, dtype=np.float32)
        data[self.player[0], self.player[1], 2] = 1
        data[self.end[0], self.end[1], 1] = 1
        return data

    def data_collapsed(self):
        d = self.data_n_dim()
        out_data = np.zeros((self.height, self.width))
        for layer in range(self._num_categories):
            out_data += d[:,:,layer] * (layer + 1)

        annotations = []
        for i in range(1, self._num_categories + 1):
            matches = np.flip(np.array(np.where(out_data == i)).T,1)
            for m in matches:
                annotations.append((self.symbols[i-1], m))

        return out_data, annotations

    def data_as_one_hot(self):
        ret_data = self.data2d().flatten()
        ret_data = self.convert_to_one_hot(ret_data)
        return ret_data

    def convert_to_one_hot(self, np_arr):
        n_values = self._num_categories + 1
        ret_array = np.eye(n_values, dtype=np.float32)[np_arr.astype(np.int32)]
        return ret_array


    def labels(self):
        labels = [self.get_dist(self.player, self.end)/self.diag_dist,
                self.angle(self.player, self.end)
                ]
        return labels
        #labels.extend(self.player.tolist())
        #labels.extend(self.end.tolist())

    def xy_data(self):
        out = np.array((self.player/max(self.width, self.height), self.end/max(self.width, self.height)))
        return out.flatten().tolist()

    def get_index_from_xy(self, xy):
        index = int((xy[1] * self.width) + xy[0])
        return index

    def get_symbol(self, in_float):
        return sorted(self.symbol_map, key=lambda s: abs(in_float - self.symbol_map[s]))[0]

    def get_indexes_within(self, m_distance, target):
        x = target[0]
        y = target[1]
        ret_list = []
        for i in range (self.width): # y
            for j in range(self.height): # x
                dist = abs(i-y) + abs(j-x)
                if dist <= m_distance and not np.array_equal((j,i), target):
                    ret_list.append((j,i))
        return ret_list

    def add_explored(self, explored):
        for ex in explored:
            if self.explored[ex[0]][ex[1]] == 0:
                self.explored[ex[0]][ex[1]] = 1


    def is_in_bounds(self, xy):
        return xy[0] >= 0 and xy[0] < self.height and xy[1] >= 0 and xy[1] < self.width


if __name__ == '__main__':
#%%
    m = Map(4,4)
    m.curriculum = 1
    m.reset()
    m.render('graphic')

    import matplotlib.pyplot as plt
#    plt.imshow(m.data())
#    plt.show()

#%%
    def human_input_test():
        #m.reset()
        m.render()
        plt.imshow(m.data())
        plt.show()
        while m.done == False:

            a = input()
            if a == "q":
                break
            n = m.action_index.index(int(a))
            s_, r, done, info = m.step(int(n))
            print("-------",r, done, info)
            m.render()
            plt.imshow(m.data())
            plt.show()

