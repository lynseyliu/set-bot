# set-bot

Plays [Set](https://en.wikipedia.org/wiki/Set_(card_game)) using a strategically-angled laptop webcam to see the board.
Cards are classified using a ConvNet, written in PyTorch.

### Usage
`python main.py -play` to play Set against the bot.  
`python main.py -check` to use the bot to check for no sets on the board.

### Dependencies
```
numpy==1.16.0
opencv-python==3.4.3.18
torch==0.4.1
torchvision==0.2.1
```
