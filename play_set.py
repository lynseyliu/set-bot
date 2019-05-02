import time

class Card:
    # color: red | green | purple
    # shape: pill | diamond | snake
    # fill: empty | solid | striped
    # number: 1 | 2 | 3
    def __init__(self, color, shape, fill, number):
        self.color = color
        self.shape = shape
        self.fill = fill
        self.number = number

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.color + " " + self.shape + " " + self.fill + " " + str(self.number)

# Example set board addition
# cards = []
# cards.append(Card('red', 'snake', 'striped', 1))

# Test all 81 cards
# for color in ['red', 'green', 'purple']:
#     for shape in ['pill', 'diamond', 'snake']:
#         for fill in ['empty', 'solid', 'striped']:
#             for number in [1, 2, 3]:
#                 cards.append(Card(color, shape, fill, number))

def find_sets(cards):
    start = time.time()
    count = 0
    results = []
    indices = []
    for i in range(len(cards)):
        c1 = cards[i]
        for j in range(i+1, len(cards)):
            c2 = cards[j]
            for k in range(j+1, len(cards)):
                count += 1
                c3 = cards[k]
                if isSet(c1, c2, c3):
                    results.append([c1, c2, c3])
                    indices.append([i, j, k])
    end = time.time()
    # print('time', end - start)
    return results, indices

def is_set(c1, c2, c3):
    return color(c1, c2, c3) and shape(c1, c2, c3) and fill(c1, c2, c3) and number(c1, c2, c3)

def color(c1, c2, c3):
    return (c1.color == c2.color == c3.color) or (c1.color != c2.color and c1.color != c3.color and c2.color != c3.color)

def shape(c1, c2, c3):
    return (c1.shape == c2.shape == c3.shape) or (c1.shape != c2.shape and c1.shape != c3.shape and c2.shape != c3.shape)

def fill(c1, c2, c3):
    return (c1.fill == c2.fill == c3.fill) or (c1.fill != c2.fill and c1.fill != c3.fill and c2.fill != c3.fill)

def number(c1, c2, c3):
    return (c1.number == c2.number == c3.number) or (c1.number != c2.number and c1.number != c3.number and c2.number != c3.number)

# sets, indices = findSets(cards)
# print('number of sets: %d' % len(indices))
# print(sets)
# print(indices)
