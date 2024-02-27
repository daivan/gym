from stable_baselines3 import *
from environments.WoodBlockPuzzle import WoodBlockPuzzle
import keyboard

env = WoodBlockPuzzle('human')

observation = env.reset()
observation = observation[0]
done = False


action = input("Lets go! Press Enter to start")

while not done:

    action = input("a: left, W: up, S: down, D: right then Enter")
    if (action == 'w'):
        action = 3
    elif (action == 'd'):
        action = 0
    elif (action == 's'):
        action = 1
    else:
        action = 2

    observation, reward, done, truncated, info = env.step(action)


env.close()