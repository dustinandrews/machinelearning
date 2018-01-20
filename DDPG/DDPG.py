# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:05:34 2017

@author: dandrews
"""
import sys
sys.path.append('D:/local/machinelearning/textmap')
from tmap import Map
import matplotlib.pyplot as plt
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("matplotlib inline")


from replay_buffer import ReplayBuffer
from actor_critic import ActorCritic
import numpy as np
from keras import backend as K
K.clear_session()
K.set_learning_phase(1)
from collections import namedtuple

class DDPG(object):
    buffer_size =               2048
    batch_size =                1024
    game_episodes_per_update =  512
    epochs = 20000
    grid_size = (10,10)
    benchmark = 1 - ((grid_size[0] + grid_size[1] - 1) * 0.02)
    input_shape = (84,84,3)
    TAU = 0.1
    min_epsilon = 0.05
    max_epsilon = 0.95
    epsilon_decay = 0.99
    reward_lambda = 0.9
    priority_replay = False
    priortize_low_scores = False
    use_maze = False
    train_actor=False
    actor_loops = 1
    solved_wins = 10 # Number of epochs with no negative scores
    use_hra = True

    def __init__(self):
        self.run_epochs = 0
        self.epochs_total = 0
        self.hybrid_loss_cumulative = []
        self.critic_loss_cumulative = []
        self.critic_target_loss_cumulative = []
        self.actor_loss_cumulative = []
        self.scores_cumulative = []
        self.critic_scores_cumulative = []
        self.actor_scores_cumulative = []
        self.winratio_cumulative = []
        self.epsilon_cumulative = []
        self.epsilon = 0.9
        self.last_lr_change = 0

        e = Map(self.grid_size[0],self.grid_size[1])
        e.USE_MAZE = self.use_maze
        e.curriculum = 1 # distance from goal player spawns at most
        self.environment = e
        self.action_count =  e.action_space.n
        self.action_shape = (self.action_count,)
        self.buffer = ReplayBuffer(self.buffer_size)
        num_rewards = len(e.hybrid_rewards())
        self.actor_critic = ActorCritic(self.input_shape, self.action_shape, num_rewards)
        self.actor_critic_target = ActorCritic(self.input_shape, self.action_shape, num_rewards)
        self.possible_actions = np.eye(e.action_space.n)[np.arange(e.action_space.n)]


    def train(self, epochs_per_plot=20):
        self.epochs_total = self.epochs + self.run_epochs
        for i in range(self.epochs):
            self.add_replays_to_buffer()
            critic_loss, actor_loss, hybrid_loss = [],[],[]

            # Approximately sample entire replay buffer
            iterations = self.buffer_size // self.batch_size
            for _ in range(iterations):
                buffer = self.buffer.sample_batch(self.batch_size)
                c_loss = self.train_critic_from_buffer(buffer)
                if self.use_hra:
                    h_loss = self.train_hybrid_from_buffer(buffer)
                    hybrid_loss.extend(h_loss)

                critic_loss.extend(c_loss)

                if self.train_actor:
                    a_loss = self.train_actor_from_buffer(buffer)
                    actor_loss.append(a_loss)

            self.actor_critic.target_train(self.actor_critic_target)

            if self.train_actor:
                a_scores = self.run_sample_games(100, use_critic=False)
                self.actor_scores_cumulative.extend(a_scores)
                self.actor_loss_cumulative.append(np.mean(actor_loss))

            # Test agent in 100 games
            c_scores = self.run_sample_games(100, use_critic=True)
            self.critic_scores_cumulative.extend(c_scores)
            self.critic_loss_cumulative.append(np.mean(critic_loss))
            self.hybrid_loss_cumulative.append(np.mean(hybrid_loss))

            # Calculate win/loss ration
            loss = (len(c_scores[c_scores <= 0]))
            win = len(c_scores[c_scores > 0])
            winratio = win / (loss+win+1e-10)
            self.winratio_cumulative.append(winratio)

            self.epsilon_cumulative.append(self.epsilon)
            self.epsilon = 0.99 - winratio # If set to 1 the agent will never play

            self.run_epochs += 1
            if self.run_epochs % epochs_per_plot == 0:
                self.plot_data("Epoch {}/{} of this run".format(i, self.epochs))
            print (self.run_epochs, end=", ")

            if self.is_solved():
                self.plot_data("Done {}".format(i))
                print("\n*********game solved at epoch {}************".format(self.run_epochs))
                break


    def play_one_session(self, random_data=False, use_critic=False):
        e = self.environment
        e.reset()
        moves = []

        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        if self.epsilon > self.max_epsilon:
            self.epsilon = self.max_epsilon

        # In case the agent hasn't had any plays yet get it one for sure
        if np.isnan(self.epsilon):
            self.epsilon = 0.9
            agent_play = True
        elif np.random.rand() > self.epsilon:
            agent_play = True
        else:
            agent_play = False

        # Rollouts were 100% agent or 100% random
        # For larger grids test mixed games
        while not e.done:
            s = e.data()

            if agent_play:
                a = self.get_action()
                action = np.argmax(a)
            else:
                # replace the agents action at random epsilon% of the time
                action = np.random.randint(self.action_count)
                a = self.possible_actions[action]

            s_, r, t, info = e.step(action)
            h = e.hybrid_rewards()
            move = namedtuple('move', ['s','a','r', 't','s_','h'])
            (move.s, move.a, move.s_, move.t, move.h) = s, a, s_, t, h
            moves.append(move)

        moves.reverse()
        r = e.cumulative_score
        for move in moves:
            move.r = r
            r *= self.reward_lambda

        moves.reverse()
        return moves, e.cumulative_score


    def add_replays_to_buffer(self):
        """
        Fills an empty buffer or adds one batch to existing buffer
        """
        rewards = []
        num = 0
        while num < self.game_episodes_per_update or self.buffer_size > self.buffer.count:
            scored_moves, reward = self.play_one_session()
            rewards.append(reward)
            for move in scored_moves:
                self.buffer.add(move.s, move.a, [move.r], [move.t], move.s_, move.h)
#            if num % 1000 == 0 and num > self.game_episodes_per_update:
            num += len(scored_moves)
#        print("Buffer status {}/{}".format(self.buffer.count, self.buffer_size))

        if self.priority_replay or self.priortize_low_scores:
            s,a,r,t,s_,h = self.buffer.to_batches()
            r = r.squeeze()
            priorities = np.zeros_like(r)
            # Adjust priorities by unpexpected Q and/or low scores
            if self.priority_replay:
                q = self.actor_critic_target.critic.predict([s,a]).squeeze()
                #q = self.actor_critic.critic.predict([s,a]).squeeze()
                priorities = np.abs(q-r)
            if self.priortize_low_scores:
                    priorities -= r
            self.buffer.set_sample_weights(-priorities)
        return rewards

    def train_critic_from_buffer(self, buffer):
        s_batch, a_batch, r_batch, t_batch, s2_batch, h_batch = buffer
        loss = self.actor_critic.train_critic(s_batch, a_batch, r_batch)
        return [loss]

    def train_hybrid_from_buffer(self, buffer):
        s_batch, a_batch, r_batch, t_batch, s2_batch, h_batch = buffer
        loss = self.actor_critic.train_hybrid(s_batch, a_batch, h_batch)
        return [loss]

    def train_actor_from_buffer(self, buffer: ReplayBuffer):
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer
        loss = self.actor_critic.train_actor(s_batch, a_batch)
        return loss

    def is_solved(self):
        winratio = self.winratio_cumulative
        if len(winratio) > self.solved_wins and np.min(winratio[-self.solved_wins:]) > 0.9998:
                    return True
        return False


    def plot_data(self, title = ""):
        ipython.magic("matplotlib inline")
        title_header = """
Input: {}, Prioritize Bad Q {}, Prioritize Score: {}
Buffer Size: {}, Batch Size: {}, rpe: {}
Use Hybrid Rewards: {} Curriculum: {}""".format(
        self.grid_size,
        self.priority_replay,
        self.priortize_low_scores,
        self.buffer_size,
        self.batch_size,
        self.game_episodes_per_update,
        self.use_hra,
        self.environment.curriculum
        )

        title = title + title_header

        fig, ax = plt.subplots(2,2, figsize=(10, 10))
        ax1 = ax[0,0]
        ax2 = ax[0,1]
        ax3 = ax[1,0]
        ax4 = ax[1,1]
        fig.suptitle(title)

        ax1.set_ylim(ymax=1.1, ymin=0)
        ax1.plot(self.epsilon_cumulative, 'r', label="Epsilon")
        ax1.plot(self.winratio_cumulative, label='moving avg win ratio')
        ax1.legend()

        smoothing = (len(self.critic_scores_cumulative) // 100 )+1
        ax2.axhline(self.benchmark, color='r', label="Solve Score")
        ax2.axhline(0.0, label="0.0")
        if self.train_actor:
            ax2.plot(self.running_mean(self.actor_scores_cumulative,smoothing), label='agent scores')
        ax2.plot(self.running_mean(self.critic_scores_cumulative,smoothing), color='orange', label='critic scores')
        ax2.legend()

        ax3.set_yscale('log')
        ax3.axhline(0, color='r')
        ax3.axhline(1, color='r', label="1.0")
        ax3.plot(self.critic_loss_cumulative, label="critic loss")
        ax3.legend()

        ax4.set_yscale('log')
        ax4.axhline(0, color='r')
        ax4.axhline(1, color='b', label='1.0')
        if self.train_actor:
            ax4.plot(self.actor_loss_cumulative, label="actor metric")
        if self.use_hra:
            ax4.plot(self.hybrid_loss_cumulative, color='green', label="hybrid loss")
        ax4.legend()

        plt.show()

    def get_action(self):
        state = np.array(self.environment.data())
        state = np.expand_dims(state, axis=0)
        if self.train_actor:
            action = self.actor_critic_target.actor.predict(state)[0]
        else:
            action = self.get_best_action_by_q()
        return action

    def get_best_action_by_q(self):
        s = self.environment.data()
        s1 = np.expand_dims(s, axis=0)
        s4 = np.repeat(s1, self.action_shape[0], axis=0)
        pred = ddpg.actor_critic_target.critic.predict([s4,self.possible_actions])
        return self.possible_actions[np.argmax(pred)]

    def run_sample_games(self, num_games = 100, egreedy=True, use_critic=False, stop_on_loss=False):
        scores = []
        e = self.environment
        for i in range(num_games):
            s = e.reset()
            while not e.done:
                if use_critic:
                    choice = np.argmax(self.get_best_action_by_q())
                else:
                    s1 = s.reshape(((1,) + s.shape))
                    pred = self.actor_critic.actor.predict(s1)[0]
                    if egreedy:
                        choice = np.argmax(pred)
                    else:
                        choice = np.random.choice(len(pred), p = pred)
                s, r, done, info = e.step(choice)
            scores.append(r)
            if r < 0 and stop_on_loss:
                print("loss detected")
                break
        return np.array(scores)

    def running_mean(self, x, N: int):
        N = int(N)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def softmax(self, a):
        a -= np.min(a)
        a = np.exp(a)
        a /= np.sum(a)
        return a

#%%


###################################################
if __name__ == '__main__':
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
#%%

    np.set_printoptions(suppress=True)
    ddpg = DDPG()
    #scores = ddpg.add_replays_to_buffer(random_data=True)

#%%


    def show_turn(e, title, index, egreedy, save):
        #plt.imshow(e.data())
        title= '\nE-greedy: {}'.format(title, e.moves, e.last_action['name'] ,str(e.player),egreedy)
        annotations = e.render_plot()

        if save:
            dirname = 'gifs/{}'.format(title)
            plt.savefig('{}fig-frame{}'.format(dirname,str(index).zfill(2)))
            plt.close()
        else:
            plt.show()
        return annotations
#%%
    def agent_play(ddpg,
                   title="",
                   egreedy=True,
                   random_agent=False,
                   save=False,
                   use_critic=True,
                   frame_pause=0.35):
        from IPython import get_ipython
        ipython = get_ipython()

        e = ddpg.environment
        s = e.reset()
        ddpg.start_state = e.data()
        ann = None
        index = 0
        plt.close()

        figManager = plt.get_current_fig_manager()
        if not 'qt5' in str(figManager):
            ipython.magic("matplotlib qt5")

        while True:
            annotations = show_turn(e, title, index, egreedy, save)
            plt.show()
            #If not inline, bring to front.
            if index == 0:
                plt.pause(1e-9)
                fig = plt.gcf()
                fig.canvas.manager.window.showMinimized()
                fig.canvas.manager.window.showNormal()
                plt.pause(frame_pause)
            plt.pause(frame_pause)

            index += 1
            for ann in annotations:
                ann.remove()
            s1 = s.reshape(((1,) + s.shape))

            if use_critic:
                s2 = np.repeat([e.data()], ddpg.action_shape[0], axis=0)
                pred = ddpg.actor_critic.critic.predict([s2, ddpg.possible_actions]).squeeze()
            else:
                pred = ddpg.actor_critic.actor.predict(s1).squeeze()

            pred = ddpg.softmax(pred)

            if egreedy:
                choice = np.argmax(pred)
            else:
                choice = np.random.choice(len(pred), p = pred)

            if random_agent:
                choice = np.random.choice(len(pred))

            s, r, done, info = e.step(choice)

            if e.done:
                ann = show_turn(e, title, index, egreedy, save)
                plt.show()
                plt.pause(frame_pause * 2)
                break
        ipython.magic("matplotlib inline")
        return e.cumulative_score, e.found_exit


#%%
    def avg_game_score(ddpg, num_games = 100, egreedy=True, use_critic=False, stop_on_loss=False):
        scores = []
        game_len = []
        e = ddpg.environment
        for i in range(100):
            s = e.reset()
            j = 0
            while not e.done:
                if use_critic:
                    choice = np.argmax(ddpg.get_best_action_by_q(ddpg))
                else:
                    s1 = s.reshape(((1,) + s.shape))
                    pred = ddpg.actor_critic.actor.predict(s1)[0]
                    if egreedy:
                        choice = np.argmax(pred)
                    else:
                        choice = np.random.choice(len(pred), p = pred)
                s, r, done, info = e.step(choice)
                j += 1
            scores.append(r)
            game_len.append(j)
            if r < 0 and stop_on_loss:
                print("loss detected")
                break
        return scores, game_len

#%%
    def compare_a_to_c(ddpg):
        e = ddpg.environment
        e.reset()
        while not e.done:
            s2 = np.array([e.data(), e.data(), e.data(), e.data()])
            apred = ddpg.actor.predict(np.array([e.data()]))
            cpred = ddpg.critic_target.predict([s2, ddpg.possible_actions]).reshape(1,4)
            cchoice = np.argmax(cpred)
            achoice = np.argmax(apred)
            #if cchoice != achoice:
            e.render()
            print("actor", apred, e._actions[e.action_index[achoice]])
            print("critic", cpred, e._actions[e.action_index[cchoice]])
            print()
            #    break
            #else:
            e.step(achoice)
        print(e.cumulative_score, e.found_exit)

#%%
    def run_n_tests(n, buffer_size = 2048, batch_size= 512, game_episodes_per_update = 256, q = True, s = True):
        winrates = []
        for i in range(n):
            print("{}/{} - buff: {}, batch: {}, epu: {}".format(i+1,n,buffer_size, batch_size, game_episodes_per_update))
            ddpg.grid_size = (4,4,3)
            ddpg.__init__()
            ddpg.epochs      =               1000
            ddpg.buffer_size =               buffer_size
            ddpg.batch_size  =               batch_size
            ddpg.game_episodes_per_update =  game_episodes_per_update
            ddpg.priority_replay          =  q
            ddpg.priortize_low_scores     =  s
            epochs, winrate = ddpg.train(epochs_per_plot=ddpg.epochs+1)
            winrates.append(winrate)

        data = {'buffer_size':ddpg.buffer_size,
                'batch_size':ddpg.batch_size,
                'game_episodes_per_update':ddpg.game_episodes_per_update,
                'prioritize bad Q': q,
                'prioritize low score': s,
                'winrates':winrates
                }
        return data


#%%
    def compare_hyperparams():
        test_results = []
        index = 0

#        for bu in range(10,13):
#            buffer_size = 2 ** bu
#            for ba in range(9,10):
#                if bu < ba:
#                    break
#                batch_size = 2 ** ba
#                for epu in range(7,10):
#                    if(ba < epu):
#                        break
#                    game_episodes_per_update = 2 ** epu
        for q in [True]:
            for s in [True]:
                    print(index)
                    index += 1
                    data = run_n_tests(10, q=q, s=s )
                    test_results.append(data)
        return test_results

#%%
    ddpg.train()
    #data = compare_hyperparams()
    #print(data)


