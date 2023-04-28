from custom_breakout_with_stacking_images import CustomBreakoutWithStackingImages


# Create an instance of your custom Breakout environment
env = CustomBreakoutWithStackingImages()

# Modify your code to work with the custom environment
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done, terminated, info = env.step(action)
    # Do something with the next state, reward, and done flag
    state = next_state
