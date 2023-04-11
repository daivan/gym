# We use conda
```
conda create --name gymnasium
conda activate gymnasium
```
Then we install everything we need
```
pip install gymnasium
pip install gymnasium[box2d]
pip install torch torchvision torchaudio
pip install stable-baselines3[extra] protobuf==3.20.*
pip install mss pydirectinput
pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support # Installing this because stable_baseline 1.7.0 has trouble with gymnasium 0.28.1. https://github.com/DLR-RM/stable-baselines3/pull/780
```

# Activate tensorboard
to run the webserver for tensorboard go to the program that is currently logging and run this command
```
tensorboard --logdir=logs
```
Make sure that port 6006 is not in use. If it is, you can change the port with the --port flag.

Where logs is the folder where the logs are currently generated.

It will prompt you with this
```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.12.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

# Start the program with a specific model
Lets say that you have trained the model and you want to test it. Make sure you have a `run_model.py`
```
python run_model.py
```