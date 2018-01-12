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
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import keras
import numpy as np
from keras import backend as K
K.clear_session()
K.set_learning_phase(1)
from collections import namedtuple

class DDPG(object):
    buffer_size =               2048
    batch_size =                512
    game_episodes_per_update =  256
    epochs = 100000
    input_shape = (2,2,1)
    benchmark = 1 - ((input_shape[0] + input_shape[1] - 1) * 0.01)
    TAU = 0.1
    min_epsilon = 0.05
    max_epsilon = 0.95
    epsilon_decay = 0.99
    reward_lambda = 0.9
    priority_replay = True
    priortize_low_scores = True

    def __init__(self):
        self.run_epochs = 0
        self.epochs_total = 0
        self.critic_loss_cumulative = []
        self.critic_target_loss_cumulative = []
        self.actor_loss_cumulative = []
        self.scores_cumulative = []
        self.agent_scores_cumulative = []
        self.winratio_cumulative = []
        self.epsilon_cumulative = []
        self.epsilon = 0.9
        self.last_lr_change = 0

        e = Map(self.input_shape[0],self.input_shape[1])
        e.curriculum = 1 # distance from goal player spawns at most
        self.environment = e
        self.action_count =  e.action_space.n
        self.output_shape = (self.action_count,)
        self.critic_output_shape = (1,)
        self.buffer = ReplayBuffer(self.buffer_size)

        cn = CriticNetwork()

        # save critic inputs for actor train
        cn.create_critic_network(self.input_shape, self.output_shape, self.critic_output_shape)

        cn.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.critic_output_shape
                )

        self.critic_model = cn.critic_model
        self.critic_target = cn.critic_target_model
        self.critic_state_input = cn.state_input
        self.critic_action_input = cn.action_input


        self.actor_network = ActorNetwork(self.input_shape, self.output_shape, cn)
        self.actor_model = self.actor_network.actor_model
        self.actor_target = self.actor_network.actor_target_model

        self.possible_actions = np.eye(e.action_space.n)[np.arange(e.action_space.n)]


    def target_train(self, source: keras.models.Model, target: keras.models.Model):
        """
        Nudges target model towards source values
        """
        tau = self.TAU
        source_weights = np.array(source.get_weights())
        target_weights = np.array(target.get_weights())
        new_weights = tau * source_weights + (1 - tau) * target_weights
        target.set_weights(new_weights)

    #TODO enable critic play for comaprisons
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
            a = self.get_action(random_data)
            action = np.argmax(a)

            if not agent_play:
                # replace the agents action at random epsilon% of the time
                action = np.random.randint(self.output_shape[0])
                a = self.possible_actions[action]

            s_, r, t, info = e.step(action)
            move = namedtuple('move', ['s','a','r', 't','s_'])
            (move.s, move.a, move.s_, move.t) = s, a, s_, t
            moves.append(move)

        moves.reverse()
        r = e.cumulative_score
        for move in moves:
            move.r = r
            r *= self.reward_lambda

        moves.reverse()
        if agent_play:
                self.agent_scores_cumulative.append(e.cumulative_score)
        return moves, e.cumulative_score


    def add_replays_to_buffer(self, random_data=False):
        """
        Fills an empty buffer or adds one batch to existing buffer
        """
        rewards = []
        num = 0
        while num < self.game_episodes_per_update or self.buffer_size > self.buffer.count:
            scored_moves, reward = self.play_one_session(random_data)
            rewards.append(reward)
            for move in scored_moves:
                self.buffer.add(move.s, move.a, [move.r], [move.t], move.s_)
#            if num % 1000 == 0 and num > self.game_episodes_per_update:
            num += len(scored_moves)
#        print("Buffer status {}/{}".format(self.buffer.count, self.buffer_size))

        if self.priority_replay or self.priortize_low_scores:
            s,a,r,t,s_ = self.buffer.to_batches()
            r = r.squeeze()
            priorities = np.zeros_like(r)
            # Adjust priorities by unpexpected Q and/or low scores
            if self.priority_replay:
                q = self.critic_target.predict([s,a]).squeeze()
                priorities = np.abs(q-r)
            if self.priortize_low_scores:
                    priorities -= r
            self.buffer.set_sample_weights(-priorities)

        return rewards

    def train_critic_from_buffer(self, buffer: list):
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer
        loss = self.critic_model.train_on_batch([s_batch, a_batch], r_batch)
        if False:
            plt.imshow(s_batch[0])
            plt.show()
            plt.title("a: {}   r: {}".format(a_batch[0], r_batch[0]))
            plt.show()
        self.target_train(self.critic_model, self.critic_target)
        return [loss]

    def train_actor_from_buffer(self, buffer: ReplayBuffer):
        return self.actor_network.train(buffer, self.critic_state_input, self.critic_action_input)

    def train(self, train_agent=True, random_data=False, epochs_per_plot=5):
        self.epochs_total = self.epochs + self.run_epochs

        for i in range(self.epochs):
            scores = []
            s = self.add_replays_to_buffer(random_data=random_data)
            scores.append(np.mean(s))
            critic_loss, actor_loss= [],[]

            # Approximately sample entire replay buffer
            iterations = self.buffer_size // self.batch_size
            for _ in range(iterations):
                buffer = self.buffer.sample_batch(self.batch_size)
                c_loss = self.train_critic_from_buffer(buffer)
                critic_loss.extend(c_loss)

                if train_agent:
                    a_loss = self.train_actor_from_buffer(buffer)
                    actor_loss.append(a_loss)
                    self.target_train(self.actor_model, self.actor_target)

            self.run_epochs += 1
            self.critic_loss_cumulative.extend(critic_loss)
            self.scores_cumulative.extend(scores)
            self.actor_loss_cumulative.extend(actor_loss)


            last_h = np.array(self.agent_scores_cumulative[-100:])
            loss = (len(last_h[last_h <= 0]))
            win = len(last_h[last_h>0])
            winratio = win / (loss+win+1e-10)
            self.winratio_cumulative.append(winratio)

            #if self.epsilon > self.min_epsilon:
                #self.epsilon -= self.epsilon_decay
            # Min score = -1, max = +1. Lower epsilon as scores improve.
#            adjusted_score = self.agent_scores_cumulative[-100:]
#            adjusted_score = np.mean(adjusted_score) + 1
#            adjusted_score /= 2

            self.epsilon_cumulative.append(self.epsilon)
            if self.epsilon > self.min_epsilon:
                self.epsilon = 0.9 - winratio

            if self.run_epochs % epochs_per_plot == 0:
                self.plot_data("Epoch {}/{} of this run".format(i, self.epochs))
            print (self.run_epochs, end=", ")

            ## Consider the game solved if average score is high and there are no losses ##
            if  len(self.agent_scores_cumulative) > 100\
                    and np.min(self.agent_scores_cumulative[-100:]) > 0\
                    and np.mean(self.agent_scores_cumulative[-100:]) >= self.benchmark:
                print("\n*********game solved at epoch {}************".format(self.run_epochs))
                break
        self.plot_data("Done".format(i))
        print()
        return i, np.mean(self.winratio_cumulative[-100:])




    def plot_data(self, title = ""):
        ipython.magic("matplotlib inline")
        title_header = """
Input: {}, Prioritize Bad Q {}, Prioritize Score: {}
Buffer Size: {}, Batch Size: {}, rpe: {}""".format(
        self.input_shape,
        self.priority_replay,
        self.priortize_low_scores,
        self.buffer_size,
        self.batch_size,
        self.game_episodes_per_update
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


        graph_len = len(self.agent_scores_cumulative)
        smoothing = (graph_len//100) + 1
        ax2.axhline(self.benchmark, color='r', label="Solve Score")
        ax2.axhline(0.0, label="0.0")
        ax2.plot(self.running_mean(self.agent_scores_cumulative,smoothing), 'b', label=' agent scores')
        ax2.legend()

        ax3.set_yscale('log')
        ax3.axhline(0, color='r')
        ax3.axhline(1, color='r', label="1.0")
        ax3.plot(self.critic_loss_cumulative, label="critic loss")
        ax3.legend()

        ax4.plot(self.actor_loss_cumulative, label="actor metric")
        ax4.legend()

        plt.show()


    def _color_y_axis(self, ax, color):
        for t in ax.get_yticklabels():
            t.set_color(color)


    def check_and_lower_learning_rate(self):
        if len(self.critic_loss_cumulative) - self.last_lr_change > 50:
            y = self.critic_loss_cumulative[-50:]
            x = np.arange(len(y))
            fit = np.polyfit(x,y,1)
            slope = fit[0] - fit[1]
            if slope > 0.10:
                self.lower_learing_rate()
                self.last_lr_change = len(self.critic_loss_cumulative)

    def lower_learing_rate(self, scale=0.1):
        lr = K.get_value(self.critic_model.optimizer.lr)
        K.set_value(self.critic_model.optimizer.lr, lr*scale)
        lr = K.get_value(self.actor_model.optimizer.lr)
        K.set_value(self.actor_model.optimizer.lr, lr*scale)
        print("New learning rates -  Critic: {}, Actor: {} ".format(
                K.get_value(self.critic_model.optimizer.lr),
                K.get_value(self.actor_model.optimizer.lr)
                ))

    def raise_learing_rate(self):
        lr = K.get_value(self.critic_model.optimizer.lr)
        K.set_value(self.critic_model.optimizer.lr, lr/10)
        lr = K.get_value(self.actor_model.optimizer.lr)
        K.set_value(self.actor_model.optimizer.lr, lr/10)
        print("New learning rates -  Critic: {}, Actor: {} ".format(
                K.get_value(self.critic_model.optimizer.lr),
                K.get_value(self.actor_model.optimizer.lr)
                ))

    def get_action(self, random_data=False, as_max=True):

        if not random_data:
            state = np.array(self.environment.data())
            state = np.expand_dims(state, axis=0)
            action = self.actor_target.predict(state)[0]
            # maybe?
            # action = np.eye(self.action_count)[np.argmax(action)]
        else:
            action = self.possible_actions[ np.random.randint(len(self.possible_actions)) ]


        if as_max:
            action = (action == action.max()).astype(float)
        return action

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
if __name__ == '__main__':
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
                   use_critic=False,
                   frame_pause=0.5):
        from IPython import get_ipython
        ipython = get_ipython()

        e = ddpg.environment
        s = e.reset()
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
                s2 = np.repeat([e.data()], ddpg.output_shape[0], axis=0)
                pred = ddpg.critic_target.predict([s2, ddpg.possible_actions]).squeeze()
            else:
                pred = ddpg.actor_target.predict(s1).squeeze()

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
                    choice = np.argmax(get_best_action_by_q(ddpg))
                else:
                    s1 = s.reshape(((1,) + s.shape))
                    pred = ddpg.actor_target.predict(s1)[0]
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

    def get_best_action_by_q(ddpg):
        s = ddpg.environment.data()
        s1 = np.expand_dims(s, axis=0)
        s4 = np.repeat(s1, ddpg.output_shape[0], axis=0)
        pred = ddpg.critic.predict([s4,ddpg.possible_actions])
        return ddpg.possible_actions[np.argmax(pred)]

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

#    e = ddpg.environment
#    plt.imshow(ddpg.environment.data())
#    plt.title('Move {}'.format(e.moves))
#
#    plt.annotate('move',xy=e.player, arrowprops=dict(facecolor='white'))
#    plt.show()
#%%
    def train_for_cycles(num, save_gifs=False):

        for cycle in range(num):
            true_epoch = ddpg.run_epochs
            true_epoch = str(true_epoch).zfill(4)
            print("epoch {}".format(true_epoch))
            critic_loss, critic_target_loss, scores = ddpg.train()
            plt.plot(ddpg.critic_loss_cumulative, label="critic")
            plt.legend()

            if save_gifs:
                plt.savefig("loss {}".format(true_epoch))
                plt.close()
            else:
                plt.show()
                plt.hist(scores)
                plt.show()
            if save_gifs:
                for i in range(5):
                    print(i)
                    agent_play(ddpg, title='critic {} epochs #{}'.format(true_epoch, i+1), save=True, use_critic=True)
                    agent_play(ddpg, title='actor {} epochs #{}'.format(true_epoch, i+1), save=True, use_critic=False)


#%%

    def show_row(num):
        s_batch, a_batch, r_batch, t_batch, s2_batch, q_error = ddpg.buffer.get_batches_from_list(ddpg.buffer.buffer)
        p_batch = ddpg.critic.predict([s_batch, a_batch])
        mse_batch = ((r_batch - p_batch)**2)
        print("Mean MSE: {}".format(mse_batch.mean()))

        plt.imshow(s_batch[num])
        plt.show()
        batch = np.array([s_batch[num],s_batch[num],s_batch[num],s_batch[num]])
        preds = ddpg.critic.predict([batch, ddpg.possible_actions])

        for i in range(len(ddpg.possible_actions)):
            act = np.argmax(ddpg.possible_actions[i])
            act = ddpg.environment.action_index[act]
            act = ddpg.environment._actions[act]['name']
            print("{}: {:5.4} ".format(act, preds[i][0]), end="")
        print()

        mse = ((r_batch[num] - p_batch[num]) ** 2).mean(axis=0)
        print("mse: {} r: {} q: {}, done: {} ".format(
                mse,
                r_batch[num][0],
                p_batch[num][0],
                t_batch[num][0]
                )
            )
        print(a_batch[num])
        a = np.argmax(a_batch[num])
        a = ddpg.environment.action_index[a]
        a = ddpg.environment._actions[a]
        print(a)
        plt.imshow(s2_batch[num])
        plt.show()
#%%
    def run_n_tests(n, buffer_size = 2048, batch_size= 512, game_episodes_per_update = 256, q = True, s = True):
        winrates = []
        for i in range(n):
            print("{}/{} - buff: {}, batch: {}, epu: {}".format(i+1,n,buffer_size, batch_size, game_episodes_per_update))
            ddpg.input_shape = (4,4,3)
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


