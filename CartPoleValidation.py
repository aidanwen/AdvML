import gym
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import numpy as np
import random

"""
Pseudocode:

Create Random Population
Create 50 units
Run for each population
Select group with the highest score
Ready Next Group – Create 3 groups:
Top group x
Crossovers of Top group, randomly selected .8x
Mutations of the Top group ½ x
Mutations of crossovers .2x
Go to 2

Pick top 30
x = 20
Crossover:
Create x (20) pairs from top
For each weight/bias, randomly pick from 2



Network Dimensions (example):


Input: 4
Dense: 4
Dense: 4
Output: 2  (Dense)

#Weights = 4*4+4*4+4*2 = 40
#Biases = 4+4+4+2 = 14
Total: 54

"""


class CartPoleEvolutionAgent():
    def __init__(self, num_episodes = 10, goal_score = 200, eval_num = 20, top_select_ratio = 0.2, mutate_chance = 0.0001):
        self.memory = []
        self.memory_scores = []
        self.env = gym.make('CartPole-v0')
        self.num_episodes = num_episodes
        self.goal_score = goal_score
        self.eval_num = eval_num
        self.mutate_chance = mutate_chance
        self.top_select = int(eval_num * top_select_ratio)
        self.createModel()
        self.memory = self.getRandWeights(eval_num)


    def createModel(self):
        self.model = Sequential()
        self.model.add(Dense(input_dim=4, units=4))
        self.model.add(Dense(units=4))
        self.model.add(Dense(units=2)) # these don't need activation

    def getRandWeights(self,num):
        base_weights = self.preprocessWeights(self.model.get_weights())
        return [[((-2) * np.random.random_sample(layer.shape) + 1) for layer in base_weights] for iter in range(num)]

    def preprocessWeights(self,weights):
        process_weights = []
        for i in range(0,3):
            process_weights.append(weights[2*i])
        return process_weights

    def unprocessWeights(self,weights):
        unprocess_weights = []
        for i in range(0,3):
            unprocess_weights.append(weights[i])
            unprocess_weights.append(np.zeros(len(weights[i][0])))
        return unprocess_weights

    def runAgents(self):
        scores = []
        for i in range(0,self.eval_num):
            self.model.set_weights(self.unprocessWeights(self.memory[i]))
            state = self.env.reset().reshape(1,4)
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.model.predict(state.reshape(1,4)))
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            self.memory_scores.append([i, total_reward])
            scores.append(total_reward)
        return scores


    def run(self):
        scores = self.runAgents()
        mean_score = np.mean(scores)
        if mean_score > self.goal_score:
            print('Done!')
            solved = True
        else:
            print('Average score of {}, (max {})'.format(mean_score,max(scores)))



agent = CartPoleEvolutionAgent()
agent.run()
