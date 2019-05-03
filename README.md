# set-bot

Plays [Set](https://en.wikipedia.org/wiki/Set_(card_game)) using a strategically-angled laptop webcam to see the board.
Cards are classified using a ConvNet, written in PyTorch.

### Setup
Download [pre-trained model](https://drive.google.com/open?id=1pB5M9qUIJTP3wuRFCLoDK1R2GT3J2T_x) and place in a models/ directory at the root of this repository.

Install the following dependencies:
```
numpy==1.16.0
opencv-python==3.4.3.18
torch==0.4.1
torchvision==0.2.1
pyobjc==5.2
pyttsx3==2.7
```

### Usage
`python main.py -play` to play Set against the bot.  
`python main.py -check` to use the bot to check for sets on the board.
