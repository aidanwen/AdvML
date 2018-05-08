import gym
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque


class CartPoleLSTMAgent():
    def __init__(self, num_episodes = 10, goal_score = 500, len_epoch = 4, sample_size = 64, gamma = 1.0):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        self.num_episodes = num_episodes
        self.goal_score = goal_score
        self.len_epoch = len_epoch
        self.sample_size = sample_size
        self.gamma = gamma
        self.buildModel()

    def buildModel(self):
        self.model = Sequential()

        self.model.add(Dense(input_dim=4, units=24, activation = 'tanh'))
        self.model.add(Dense(units=48, activation = 'tanh'))
        self.model.add(Dense(units=2, activation='linear'))
        self.model.compile(loss='mse',optimizer=Adam(lr=0.01,decay=0.001))

    def getAction(self, state):
        return self.env.action_space.sample() if (np.random.random() <= 0.1) else np.argmax(self.model.predict(state.reshape(1,4)))

    def train(self):
        x, y = [],[]
        samples = random.sample(self.memory, self.sample_size)
        for state, reward, action, next_state, done in samples:
            y_temp = self.model.predict(state.reshape(1,4))
            y_temp[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state.reshape(1,4)))
            x.append(state)
            y.append(y_temp[0])
        x_train = np.vstack(x)
        y_train = np.vstack(y)
        self.model.fit(x_train, y_train, batch_size=len(x_train))

        # self.model.save_weights('model_weights.h5')
        # check that you are passing in the right observations. is reshape working right, etc.


    def run_episode(self):
        state = self.env.reset().reshape(1,4)
        total_reward = 0
        done = False
        while not done:
            self.env.render()
            action = self.getAction(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.memory.append((state, total_reward, action, next_state, done))
            state = next_state
        return total_reward

    def run(self):
        solved = False
        iter = 0
        while not solved:
            reward = 0
            iter += 1
            for _ in range(self.num_episodes):
                run_reward = self.run_episode()
                reward += run_reward

            if int(reward/self.num_episodes) >= self.goal_score:
                print("Solved!")
                solved = True
            else:
                print('Iteration {}, Average Reward: {}'.format(iter, reward/self.num_episodes))
                self.train()

agent = CartPoleLSTMAgent()
agent.run()
