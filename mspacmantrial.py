import gym
import sys
import pylab
import random
import os
import operator
from collections import deque

from skimage import io, color, transform

import numpy as np

import tensorflow as tf

from keras.layers import Dense,Convolution2D,Flatten,Activation
from . import utils
from . import conv_utils
from .. import backend as K
from .tensorflow_backend import *

from keras.optimizers import Adam
from keras.models import Sequential
#environment parameters
NUM_EPISODES = 80000000
MAX_TIMESTEPS =6
FRAME_SKIP =6
PHI_LENGTH = 4

#agent parameters
#NAIVE_RANDOM = 
PREPROCESS_IMAGE_DIM = 84

class DQNAgent:

    def __init__(self):

    	        # if you want to see Pacman learning, then change to True
        self.render = False
        self.load_model = False

        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-6
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.01
        self.batch_size = 500
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=5000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()


   
    def build_model(PREPROCESS_IMAGE_DIMPREPROCESS_IMAGE_DIM):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(1, self.processed_image_dim, self.processed_image_dim)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        model.add(Activation('relu'))
        model.add(Dense(9,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
   
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        print("We finish building the model")
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def append_sample(self, state, action, reward, next_state, done):
    	"""
        Add an experience replay example to our agent's replay memory. If
        memory is full, overwrite previous examples, starting with the oldest
        """
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay        
    
        
    def preprocess_observation(self, observation, prediction=False):
        """
        Helper function for preprocessing an observation for consumption by our
        deep learning network
        """
        grayscale_observation = color.rgb2gray(observation)
        resized_observation = transform.resize(grayscale_observation, (1, self.processed_image_dim, self.processed_image_dim)).astype('float32')
        if prediction:
            resized_observation = np.expand_dims(resized_observation, 0)
        return resized_observation
        
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

def preprocess_observation(observation):
    """
    preprocesses a given observation following the steps described in the paper
    """
    grayscale_observation = color.rgb2gray(observation)
    resized_observation = transform.resize(grayscale_observation, (PREPROCESS_IMAGE_DIM, PREPROCESS_IMAGE_DIM)).astype('float32')
    return resized_observation

if __name__ == "__main__":
    """
    Entry-point for running Ms. Pac-man simulation
    """
    env = gym.make('MsPacman-v0')

    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent()
    
    #initialize auxiliary data structures
    state_list = [] 
    tot_frames = 0

    for i_episode in range(NUM_EPISODES):
        print ("Episode: %s" % i_episode)


        

        while True:

            
            
            #ensure state list is populated
            if tot_frames < PHI_LENGTH:
                state_list.append(preprocess_observation(observation))
                tot_frames += 1
                continue



                #update state list with next observation
                state_list.append(preprocess_observation(observation))
                state_list.pop(0)






