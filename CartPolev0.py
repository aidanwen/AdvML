import gym
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

class CartPoleLSTMAgent():
    def __init__(self, num_episodes, goal_score, goal_accuracy):
        self.env = gym.make('CartPole-v0')
        self.num_episodes = num_episodes
        self.goal_score = goal_score
        self.goal_accuracy = goal_accuracy
        self.buildModel()

    def buildModel(self):
        self.model = Sequential()

        self.model.add(LSTM(input_dim=4, units=32, return_sequences=True))
        self.model.add(LSTM(units=32))
        self.model.add(Dense(units=2, activation='linear'))
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])

    def getAction(self, input):
        if self.accuracy > self.goal_accuracy:
            return self.model.predict_on_batch(input)
        else:
            return self.env.action_space.sample()

    def train(self):

    def run(self):
        for episode in range(self.num_episodes):
            action = self.env.action_space.sample()
            self.env.reset()
            total_reward = 0
            while not done:
                env.render()
                observation, reward, done, info = self.env.step(action)
                action = self.getAction(np.reshape(observation,[1,4]))
                total_reward += reward


            if total_reward >= self.goal_score:
                print("Solved!")
            else:
                self.train()
