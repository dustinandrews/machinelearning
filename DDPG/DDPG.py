# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:05:34 2017

@author: dandrews
"""
import sys
sys.path.append('D:/local/machinelearning/textmap')
from tmap import Map
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import keras
import numpy as np
from keras import backend as K
K.clear_session()
K.set_learning_phase(1)

class DDPG(object):
    buffer_size = 1000
    batch_size = 50
    epochs = 1000
    input_shape = (5,5,3)
    decay = 0.9
    TAU = 0.1
    critic_loss_cumulative = []    
    critic_target_loss_cumulative = []
    actor_loss_cumulative = []
    scores_cumulative = []
    run_epochs = 0
    epsilon = 0.3
    decay = 0.9999
    


    def __init__(self):
        e = Map(self.input_shape[0],self.input_shape[1])
        self.environment = e
        self.output_shape = (e.action_space.n,)
        self.critic_output_shape = (1,)
        self.buffer = ReplayBuffer(self.buffer_size)

        cn = CriticNetwork()
        
        # save critic inputs for actor train
        self.critic_state_input, self.critic_action_input, self.critic =\
            cn.create_critic_network(self.input_shape, self.output_shape, self.critic_output_shape)
        
        _, _, self.critic_target = cn.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.critic_output_shape
                )

        self.actor_network = ActorNetwork(self.input_shape, self.output_shape, self.critic)
        self.actor = self.actor_network.actor_model
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

    def fill_replay_buffer(self, random_data=False):
        e = self.environment
        rewards = []
        for i in range(self.buffer_size):
            if e.done:
                e.reset()
            a = self.get_action(random_data)
            s = e.data()
            
            if np.random.rand() < self.epsilon:
                action = np.random.randint(len(a))
            else:
                action = np.argmax(a)
                self.epsilon *= self.decay            
            (s_, r, t, info) = e.step(action)
            self.buffer.add(s, a, [r], [t], s_)
            rewards.append(r)
        return rewards

    def train_critic_from_buffer(self, buffer):        
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer
        #print("[s_batch {}, a_batch {}], rbatch {}".format(s_batch.shape, a_batch.shape, r_batch.shape))
        loss = self.critic.train_on_batch([s_batch, a_batch], r_batch)            
        self.target_train(self.critic, self.critic_target)
        return [loss]

    def train_actor_from_buffer(self, buffer):        
        return self.actor_network.train(buffer, self.critic_state_input, self.critic_action_input)        

    def train(self):
        random_data = True
        critic_loss, critic_target_loss, actor_loss, scores= [],[],[],[]
        #last_lr_change = 0
        for i in range(self.epochs):
            s = self.fill_replay_buffer(random_data=random_data)
            buffer = self.buffer.sample_batch(self.buffer.buffer_size)

            scores.extend(s)
            for _ in range(self.batch_size):
                c_loss = self.train_critic_from_buffer(buffer)
                #ct_loss = self.get_loss_from_buffer(self.critic_target)
                a_loss = self.train_actor_from_buffer(self.buffer)
                critic_loss.extend(c_loss)
                actor_loss.append(a_loss)
            #critic_target_loss.extend(ct_loss)
            random_data = False
            print("epoch {}/{}".format(i+1, self.epochs))

            self.run_epochs += 1
            # change LR            
            self.critic_loss_cumulative.extend(critic_loss)
            self.critic_target_loss_cumulative.extend(critic_target_loss)
            self.scores_cumulative.extend(scores)
            self.actor_loss_cumulative.extend(actor_loss)
            plt.plot(self.critic_loss_cumulative, label="critic_loss")
            plt.plot(self.actor_loss_cumulative, label="actor loss")
            plt.legend()
            plt.show()
            print()


    def check_and_lower_learning_rate(self, i, last_lr_change, critic_loss, c_loss):
        if i - last_lr_change > 200:
                mean_loss = np.mean(critic_loss[-50])
                #print(i, mean_loss, end=", ")
                if np.mean(c_loss) >= mean_loss:
                    lr = K.get_value(self.critic.optimizer.lr)
                    print("Lowering Learning rate {} by order of magnitude.".format(lr))
                    K.set_value(self.critic.optimizer.lr, lr/10)
                    last_lr_change = i
    
    def lower_learing_rate(self):
        lr = K.get_value(self.critic.optimizer.lr)        
        K.set_value(self.critic.optimizer.lr, lr/10)
        print(K.get_value(self.critic.optimizer.lr))


    def get_loss_from_buffer(self, model: keras.models.Model):
        s_batch, a_batch, r_batch, t_batch, s2_batch  = self.buffer.sample_batch(self.batch_size)
        pred = model.predict([s_batch, a_batch])
        delta = np.square(pred - r_batch)
        return delta


    def get_action(self, random_data=False):
        if not random_data:
            state = np.array(self.environment.data())
            state = np.expand_dims(state, axis=0)
            action = self.actor_target.predict(state)[0]                        
        else:
            action = self.possible_actions[ np.random.randint(len(self.possible_actions)) ]
        return action

    def running_mean(self, x, N: int):
        N = int(N)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    def softmax(self, a):
        sm = np.exp(a * 1e-5)
        sm /= np.sum(sm)
        return sm

#%%
if __name__ == '__main__':
#%%
    
    np.set_printoptions(suppress=True)
    ddpg = DDPG()
    ddpg.fill_replay_buffer(random_data=True)

#%%
        
        
    def show_turn(e, title, index, egreedy, save):
        plt.imshow(e.data())
        plt.title('{}  Turn: {}  Move: {} to {}\nE-greedy: {}'.format(title, e.moves, e.last_action['name'] ,str(e.player),egreedy))
        startpos = e.player - np.array(e.last_action['delta']) * 0.5
        lastpos = e.player +  np.array(e.last_action['delta']) * 0.5
        ann = plt.annotate('',xytext=startpos[::-1], xy=lastpos[::-1], arrowprops=dict(facecolor='white'))
        plt.axis('off')
        if save:
            dirname = 'gifs/{}'.format(title)
            plt.savefig('{}fig-frame{}'.format(dirname,str(index).zfill(2)))
            plt.close()
        else:
            plt.show()
            plt.pause(0.2)        
        return ann
#%%        
    def agent_play(ddpg, title="", egreedy=True, random_agent=False, save=False, use_critic=False):
        
        e = ddpg.environment
        s = e.reset()
        ann = None
        index = 0        
        plt.close()
        actions = np.eye(4)[np.arange(4)]
        while True:                  
            ann = show_turn(e, title, index, egreedy, save)
            index += 1
            if ann:
                ann.remove()            
            
            s1 = s.reshape(((1,) + s.shape))
            #pred = ddpg.actor_target.predict(s1).squeeze()
            
            if use_critic:            
                s2 = np.array([e.data(), e.data(), e.data(), e.data()])
                pred = ddpg.critic_target.predict([s2, actions.squeeze()]).squeeze()
                pred -= np.min(pred)
                pred = np.exp(pred)
                pred /= np.sum(pred)
            else:            
                pred = ddpg.actor.predict(s1).squeeze()
            
            if egreedy:
                choice = np.argmax(pred)
            else:
                choice = np.random.choice(len(pred), p = pred)

            if random_agent:
                choice = np.random.choice(len(pred))
            s, r, done, info = e.step(choice)
            
            if e.done:
                ann = show_turn(e, title, index, egreedy, save)
                break
        return e.cumulative_score, e.found_exit
        

#%%
    def avg_game_score(ddpg, num_games = 100, egreedy=True, use_critic=False):
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
            scores.append(e.cumulative_score)
            game_len.append(j)
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
        e = Map(ddpg.input_shape[0],ddpg.input_shape[1])        
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
    s_batch, a_batch, r_batch, t_batch, s2_batch = ddpg.buffer.sample_batch(len(ddpg.buffer.buffer))
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
    print("mse: {:5.2} r: {:5.3} q: {:5.3}".format(mse, r_batch[num][0], p_batch[num][0]))
    a = np.argmax(a_batch[num])
    a = ddpg.environment.action_index[a]
    a = ddpg.environment._actions[a]
    print(a)
    plt.imshow(s2_batch[num])
    plt.show()
        