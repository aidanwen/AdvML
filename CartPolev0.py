import gym
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

class CartPoleLSTMAgent():
    def __init__(self, num_episodes, goal_score):
        pass

    def buildModel(self):
        self.model = Sequential()

        self.model.add(LSTM(input_dim=4, units=32, return_sequences=True))
        self.model.add(LSTM(units=32))
        self.model.add()

env = gym.make('CartPole-v0')
env.reset()
env.render()
