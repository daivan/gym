import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create the Lunar Lander environment
env = gym.make('LunarLander-v2')

# Define the model architecture
model = Sequential([
    Dense(64, input_shape=(8,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action_probs = model.predict(np.array([state]))[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        next_state, reward, done, info = env.step(action)
        target = np.zeros((1, 4))
        target[0, action] = 1
        model.fit(np.array([state]), target, verbose=0)
        state = next_state
