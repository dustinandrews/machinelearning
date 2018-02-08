# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:30:17 2018

@author: dandrews

Top level library to abstract Nethack for bots that follows AIGym conventions
"""

from NhInterface import NhClient
from nhstate import NhState


class NhEnv():
    """
    More or less replicate an AI Gym environment for connecting to a NN.
    """

    is_done = False
    _action_rating = 1
    strategies = {
            0: 'direct',
            1: 'explore'
            }

    def __init__(self):
        self.nhc = NhClient()
        self.actions = self.nhc.nhdata.get_commands(1)
        self.num_actions = len(self.actions)
        self.nhstate = NhState(self.nhc)

    def __del__(self):
        self.nhc.close()

    def reset(self):
        """
        Start a new game
        """
        self.nhc.start_session()
        self.nhc.reset_game()
        self.nhc.start_session()
        return self.data()

    def connect(self):
        """
        Connect to existing game
        """
        self.nhc.start_session()

    def step(self, action: int, strategy: int):
        if self.is_done:
            raise ValueError("Simulation ended and must be reset")
        last_status = self.nhc.get_status()
        last_screen = self.nhc.buffer_to_npdata()
        if self.strategies[strategy] == 'explore':
           self. _do_exploration_move(action)
        else:
            self._do_direct_action(action)

        self.is_done = self.nhstate.check_game_state()

        #s_, r, t, info
        s_, info = self.data()
        t = self.is_done
        r = self.score_move(last_status, last_screen)
        return s_, r, t, info

    def score_move(self, last_status, last_screen):
        new_status = self.nhc.get_status()
        new_screen = self.nhc.buffer_to_npdata()
        screen_diff = last_screen - new_screen
        explore = len(screen_diff[screen_diff != 0])
        score = explore - 1 # Offset score turn and punish no-ops like wall bumps
        for key in new_status:
            if last_status[key] < new_status[key]:
                score += 1
        return score


    def _do_direct_action(self, action):
        if action >= self.num_actions:
            raise ValueError('No such action {}, limit is {}'.format(action, self.num_actions-1))
        self.nhc.send_command(action)

    def _do_exploration_move(self, action):
        if action not in self.nhc.nhdata.MOVE_COMMANDS:
            # No op
            return
        else:
           self.nhc.send_string("G")
           self.nhc.send_string(str(action))


    def data(self):
        output = []
        output.append(self.nhc.buffer_to_rgb())
        output.append(self.nhc.get_status())
        return(output)

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt

    nhe = NhEnv()
    data = nhe.connect()
    print("\n".join(nhe.nhc.screen.display))


#%%
    def random_agent(nhe):
        actions = np.arange(1,10)
        reps = 0
        while True:
            strategy = np.random.choice([0,1])
            action = np.random.choice(actions)
            s_, r, t, info = nhe.step(action, strategy)

            if reps % 100 == 0:
                print(reps)
                plt.imshow(s_)
                plt.show()
                print("action: {} strategy: {} score: {}".format(action, strategy, r))

            if reps > 0:
                x = ''
                reps -=1
                print(reps)

            if reps < 1:
                    print("==> ", end="")
                    x = input()

            if 'q' in x:
                break
            if x != '':
               try:
                   reps = int(x)
               except:
                   reps = 0

