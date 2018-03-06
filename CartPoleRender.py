import gym
env = gym.make('CartPole-v0')
env.reset()
total_reward = 0
for _ in range(1000):
    env.render()
    x = env.action_space.sample()
    observation, reward, done, info = env.step(x)
    total_reward += reward
    if done:
        print(total_reward)
        env.reset()
        total_reward = 0
