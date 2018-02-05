# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:30:17 2018

@author: dandrews

Top level library to abstract Nethack for bots that follows AIGym conventions
"""

from NhInterface import NhClient

class nhclient():
    _action_rating = 1
    strategies = {
            0: 'direct',
            1: 'explore'
            }

    def __init__(self):
        self.nhc = NhClient()
        self.actions = self.nhc.nhdata.get_commands(1)
        self.num_actions = len(self.actions)


    def reset(self):
        """
        Start a new game
        """
        self.nhc.start_session()
        self.nhc.reset_game()
        self.nhc.start_session()
        return self.data()

    def step(self, action: int, strategy: int):
        last_status = self.nhc.get_status()
        if self.strategies[strategy] == 'explore':
            _do_exploration_move(action)
        else:
            _do_direct_action(action)

        t = self.is_done()
        #s_, r, t, info
        s_ = self.data()
        r = self.score_move(last_status)
        info = self.nhc.get_status()
        return s_, r, t, info

    def score_move(self, last_status):
        new_status = self.nhc.get_status()
        score = -1 # Offset score turn and punish no-ops like wall bumps
        for key in new_status:
            if last_status[key] < new_status[key]:
                score += 1

    def _do_direct_action(self, action):
        if action >= self.num_actions:
            raise ValueError('No such action {}, limit is {}'.format(action, self.num_actions-1))
        action_num = self.actions[action]
        self.nhc.send_command(action_num)



    def _do_exploration_move(self, action):
        if action not in self.nhc.nhdata.MOVE_COMMANDS:
            # No op?
            return
        else:
           self.nhc.send_string("g" + str(action))


    def is_done(self):
        return False

    def data(self):
        output = []
        output.append(self.nhc.buffer_normalized_npdata())
        output.append(self.nhc.get_status)