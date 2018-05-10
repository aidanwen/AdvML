import gym
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
import numpy as np
import random


class CartPoleEvolutionAgent():
    def __init__(self, num_episodes = 10, goal_score = 500, eval_num = 50):
        self.memory = []
        self.env = gym.make('CartPole-v0')
        self.num_episodes = num_episodes
        self.goal_score = goal_score
        self.eval_num = eval_num
        self.getInitWeights(eval_num)

    def getInitWeights(self,num):
        self.model = Sequential()
        for i in range(num):
            self.model.add(Dense(input_dim=4, units=12, activation = 'tanh'))
            self.model.add(Dense(units=24, activation = 'tanh'))
            self.model.add(Dense(units=2, activation='linear'))

            print(self.model.get_weights())
            print(len(self.model.get_weights()))
            print(self.model.get_weights().shape())
            self.memory.append(self.preprocessWeights(self.model.get_weights()))

    def processWeights(self,weights):
        process_weights = []
        for i in range(0,3):
            process_weights.append(weights[2*i])
        return process_weights

    def unprocessWeights(self,weights):
        unprocess_weights = []
        for i in range(0,3):
            unprocess_weights.append(weights[i])
            unprocess_weights.append(numpy.zeros(weights[2*i][0].shape))
        return unprocess_weights

    def runAgents(self):
        for weights in self.memory:
            self.model.set_weights(unprocessWeights(weights))
            state = self.env.reset().reshape(1,4)
            while not done:
                self.env.render()
                action = self.model.predict(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            weights.append(total_reward)

    def evolve(self):
        select = np.sort(self.memory)[:30]
        for i in range(0,20):
            for j in range()

        for i in range(0,10):
            pass

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

    def run(self):
        solved = False
        iter = 0

        for i in range(eval_num):
            self.weights()

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

agent = CartPoleEvolutionAgent()
agent.run()
