import time

class Card:
    # color =  r | g | b
    # shape = d | o | s
    # fill = e | d | f
    # number = 1 | 2 | 3
    def __init__(self, color, shape, fill, number):
        self.color = color
        self.shape = shape
        self.fill = fill
        self.number = number

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.color + " " + self.shape + " " + self.fill + " " + str(self.number)

cards = []
cards.append(Card('r', 's', 'd', 1))
cards.append(Card('r', 'o', 'd', 2))
cards.append(Card('r', 'd', 'd', 2))
cards.append(Card('r', 'o', 'f', 2))
cards.append(Card('g', 'd', 'f', 1))
cards.append(Card('b', 'o', 'f', 3))
cards.append(Card('r', 'd', 'e', 1))
cards.append(Card('b', 'd', 'f', 2))
cards.append(Card('g', 's', 'd', 2))
cards.append(Card('g', 's', 'e', 1))
cards.append(Card('g', 'o', 'e', 3))
cards.append(Card('r', 'd', 'f', 3))

def findSets(cards):
    start = time.time()
    count = 0
    results = []
    for i in range(10):
        c1 = cards[i]
        for j in range(i+1, 12):
            c2 = cards[j]
            for k in range(j+1, 12):
                count += 1
                c3 = cards[k]
                if isSet(c1, c2, c3):
                    results.append([c1, c2, c3])
    end = time.time()
    print('time', end - start)
    return results

def isSet(c1, c2, c3):
    return color(c1, c2, c3) and shape(c1, c2, c3) and fill(c1, c2, c3) and number(c1, c2, c3)

def color(c1, c2, c3):
    return (c1.color == c2.color == c3.color) or (c1.color != c2.color and c1.color != c3.color and c2.color != c3.color)

def shape(c1, c2, c3):
    return (c1.shape == c2.shape == c3.shape) or (c1.shape != c2.shape and c1.shape != c3.shape and c2.shape != c3.shape)

def fill(c1, c2, c3):
    return (c1.fill == c2.fill == c3.fill) or (c1.fill != c2.fill and c1.fill != c3.fill and c2.fill != c3.fill)

def number(c1, c2, c3):
    return (c1.number == c2.number == c3.number) or (c1.number != c2.number and c1.number != c3.number and c2.number != c3.number)

sets = findSets(cards)
print(sets)
print(len(sets))