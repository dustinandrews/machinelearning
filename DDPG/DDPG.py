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

class DDPG(object):
    buffer_size = 1000
    batch_size = 50
    epochs = 10
    input_shape = (10,10,3)
    decay = 0.9
    TAU = 0.1
    critic_loss_cumulative = []
    actor_loss_cumulative = []
    critic_target_loss_cumulative = []
    scores_cumulative = []
    run_epochs = 0


    def __init__(self):
        e = Map(self.input_shape[0],self.input_shape[1])
        self.output_shape = e.action_space.n
        self.action_input_shape = (1,)

        self.environment = e

        self.buffer = ReplayBuffer(self.buffer_size)

        actor_network = ActorNetwork()
        self.actor = actor_network.create_actor_network(
                self.input_shape,
                self.output_shape)
        self.actor_target = actor_network.create_actor_network(
                self.input_shape,
                self.output_shape)

        critic_network = CriticNetwork()
        self.critic = critic_network.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.action_input_shape
                )
        self.critic_target = critic_network.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.action_input_shape
                )



    def step(self):
        state = np.expand_dims(self.environment.data_normalized(), axis=0)
        prediction = self.actor.predict([state],1)
        return prediction

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
            (s_, r, t, info) = e.step(a)
            self.buffer.add(s, [a], [r], [t], s_)
            rewards.append(r)
        return rewards

    def train_critic_from_buffer(self, buffer):
        loss_record = []        
        for i in range(self.batch_size):
            buffer = self.buffer.sample_batch(self.buffer_size) #randomize order, helps?
            s_batch, a_batch, r_batch, t_batch, s2_batch = buffer
            loss = self.critic.train_on_batch([s_batch, a_batch], r_batch)            
            loss_record.append(loss)
            self.target_train(self.critic, self.critic_target)
        return loss_record
    
    

    def train_actor_from_buffer(self, buffer):
        loss_record = []
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer
        a_batch = a_batch.squeeze()
        a_one_hot = np.eye(self.output_shape)[a_batch]
        critic_predictions = self.critic_target.predict([s_batch,a_batch])
        gradient  = a_one_hot * critic_predictions
        # Convert gradient to softmax from value function
        gradient -= np.min(gradient) 
        gradient_sums = np.sum(gradient, axis=1)
        gradient_sums = gradient_sums.reshape((gradient_sums.shape)+(1,))
        gradient /= gradient_sums
        for i in range(self.batch_size):
            loss = self.actor.train_on_batch(s_batch, gradient)            
            loss_record.append(loss)
            self.target_train(self.actor, self.actor_target)
        return loss_record

    def train(self):
        random_data = False
        actor_loss,critic_loss, critic_target_loss, scores= [],[],[], []
        #last_lr_change = 0
        for i in range(self.epochs):
            s = self.fill_replay_buffer(random_data=random_data)
            buffer = self.buffer.sample_batch(self.buffer.buffer_size)

            scores.extend(s)
            c_loss = self.train_critic_from_buffer(buffer)
            ct_loss = self.get_loss_from_buffer(self.critic_target)
            a_loss = self.train_actor_from_buffer(buffer)
            critic_loss.extend(c_loss)
            actor_loss.extend(a_loss)
            critic_target_loss.extend(ct_loss)
            random_data = False
            print("epoch {}/{}".format(i+1, self.epochs))
            self.run_epochs += 1
            # change LR
            
        self.critic_loss_cumulative.extend(critic_loss)
        self.actor_loss_cumulative.extend(actor_loss)
        self.critic_target_loss_cumulative.extend(critic_target_loss)
        self.scores_cumulative.extend(scores)        
        return critic_loss, actor_loss, critic_target_loss, scores

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


    def get_loss_from_buffer(self, model: keras.models.Model):
        s_batch, a_batch, r_batch, t_batch, s2_batch  = self.buffer.sample_batch(self.batch_size)
        pred = model.predict([s_batch, a_batch])
        delta = np.square(pred - r_batch)
        return delta


    def get_action(self, random_data=False):
        if not random_data:
            state = np.array(self.environment.data())
            state = np.expand_dims(state, axis=0)
            pred = self.actor_target.predict(state)
            pred = pred.squeeze()
            # e-greedy
            #action = np.argmax(pred)

            # weighted random
            action = np.random.choice(len(pred), p = pred)
        else:
            action = np.random.randint(0, self.output_shape)
        return action

    def running_mean(self, x, N: int):
        N = int(N)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

#%%
if __name__ == '__main__':
#%%
    
    np.set_printoptions(suppress=True)
    ddpg = DDPG()
    ddpg.fill_replay_buffer(random_data=True)

#%%
    def smoke_test():
        ddpg = DDPG()
        ddpg.epochs=1
        pred = ddpg.step()
        print(pred)
        ddpg.fill_replay_buffer(random_data=True)
        s_batch, a_batch, r_batch, t_batch, s2_batch = ddpg.buffer.sample_batch(10)
        critic_loss, actor_loss, critic_target_loss, scores = ddpg.train()
        running_mean_window = ddpg.batch_size//10

        cl_rm = ddpg.running_mean(critic_loss, running_mean_window)
        al_rm = ddpg.running_mean(actor_loss, running_mean_window)
        ct_rm = ddpg.running_mean(critic_target_loss, running_mean_window)

        plt.plot(cl_rm , label="critic_loss")
        plt.plot(al_rm, label="actor_loss")
        plt.plot(ct_rm, label="critic_target_loss")
        plt.legend()
        plt.show()
        plt.plot(scores, label="scores")
        plt.legend()
        plt.show()

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
        actions = np.arange(4).reshape(4,1)
        while True:                  
            ann = show_turn(e, title, index, egreedy, save)
            index += 1
            if ann:
                ann.remove()            
            
            s1 = s.reshape(((1,) + s.shape))
            #pred = ddpg.actor_target.predict(s1).squeeze()
            
            if use_critic:            
                s2 = np.array([e.data(), e.data(), e.data(), e.data()])
                pred = ddpg.critic_target.predict([s2, actions]).squeeze()
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
    def avg_game_len(ddpg, num_games = 100, egreedy=True):

        scores = []
        game_len = []
        for i in range(100):
            #action = np.random.choice(len(pred), p = pred)
            e = Map(ddpg.input_shape[0],ddpg.input_shape[1])
            s = e.reset()
            j = 0
            while not e.done:
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
    def performance_over_iterations(ddpg, num):

        cl,tcl,atl,sc,gl = [],[],[],[],[]

        for i in range(num):
            print("iteration {}/{}".format(i+1,num))
            critic_loss, critic_target_loss, actor_target_loss, scores = ddpg.train()
            cl.extend(critic_loss)
            tcl.extend(critic_target_loss)
            atl.extend(actor_target_loss)

            scores, game_len = avg_game_len(ddpg, num)
            sc.append(scores)
            gl.append(game_len)
            print("scores")
            plt.hist(scores)
            plt.show()
            print("len")
            plt.hist(game_len)
            plt.show()

        #critic_loss, critic_targert_loss, actor_target_loss, scores, game_len = performance_over_iterations(ddpg, 100)
        return cl,tcl,atl,sc,gl
 
#%%
    def compare_a_to_c(ddpg):
        e = Map(ddpg.input_shape[0],ddpg.input_shape[1])
        actions = np.arange(4).reshape(4,1)
        while not e.done:
            s2 = np.array([e.data(), e.data(), e.data(), e.data()])
            apred = ddpg.actor.predict(np.array([e.data()]))
            cpred = ddpg.critic_target.predict([s2, actions]).reshape(1,4)
            cchoice = np.argmax(cpred)
            achoice = np.argmax(apred)
            #if cchoice != achoice:
            #e.render()
            print("actor", apred, e._actions[e.action_index[achoice]])
            print("critic", cpred, e._actions[e.action_index[cchoice]])
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
    def train_for_cycles():
                   
        for cycle in range(10):
            true_epoch = (cycle + ddpg.run_epochs + 1) * 10
            true_epoch = str(true_epoch).zfill(3)
            print("cycle {}".format(cycle+1))
            critic_loss, actor_loss, critic_target_loss, scores = ddpg.train()
            plt.plot(critic_loss)
            plt.savefig("loss {}".format(true_epoch))
            plt.close()
            for i in range(5):
                print(i)
                agent_play(ddpg, title='critic {} epochs #{}'.format(true_epoch, i+1), save=True, use_critic=True)
                agent_play(ddpg, title='actor {} epochs #{}'.format(true_epoch, i+1), save=True, use_critic=False)
        
        
        
        