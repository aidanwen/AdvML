import gym
env = gym.make('CartPole-v0')
env.reset()
total_reward = 0
print(env.reset()e)
for i in range(1000):
    env.render()
    x = env.action_space.sample()
    print(x)
    observation, reward, done, info = env.step(x)
    total_reward += reward
    if done:
        print("done!")
        print(reward)
        print(total_reward)
        print(i)
        env.reset()

        total_reward = 0
