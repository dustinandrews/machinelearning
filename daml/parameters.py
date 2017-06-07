# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:48:55 2017

@author: dandrews
"""

class hyperparameters:
        
        def __init__(self,
                 hidden_dim=0,
                 optimizer= None,
                 loss = None,
                 learning_rate=1e-4, 
                 minibatch_size=100,
                 epochs=100,
                 l1reg = 0,
                 l2reg = 0,
                 stop_at_loss = 0,
                 discount = 0,
                 dropout = 0.20
                 ):
            self.hidden_dim    = hidden_dim
            self.learning_rate = learning_rate
            self.minibatch_size= minibatch_size
            self.epochs        = epochs
            self.loss          = loss 
            self.l1reg         = l1reg 
            self.l2reg         = l2reg
            self.stop_at_loss  = stop_at_loss
            self.discount      = discount
            self.optimizer     = optimizer
