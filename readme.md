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
```

