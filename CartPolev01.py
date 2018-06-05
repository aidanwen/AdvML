import gym
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

class CartPoleLSTMAgent():
    def __init__(self, num_episodes, goal_score, goal_accuracy, len_epoch = 4):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        self.num_episodes = num_episodes
        self.goal_score = goal_score
        self.goal_accuracy = goal_accuracy
        self.len_epoch = len_epoch

        self.buildModel()

    def buildModel(self):
        self.model = Sequential()

        self.model.add(LSTM(input_dim=5, units=8, return_sequences=True))
        self.model.add(LSTM(units=8))
        self.model.add(Dense(units=2, activation='linear'))
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])

    def getAction(self, state): #Q
        if self.Q(state, 0) >= self.Q(state,1):
            return 0
        else:
            return 1
        return

    def Q(self, state, action):
        return self.model.predict(state.append(action))

    def train(self):

        self.model.save_weights('model_weights.h5')


    def run_episode(self):
        action = self.env.action_space.sample()
        self.env.reset()
        total_reward = 0
        while not done:
            env.render()
            observation, reward, done, info = self.env.step(action)
            self.memory.append([observation, reward, action])
            action = self.getAction(np.reshape(observation,[1,4]))
            total_reward += reward


        if total_reward >= self.goal_score:
            print("Solved!")
        else:
            self.train()
