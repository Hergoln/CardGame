from card_game import Card
import numpy as np

ranks_encode = {'9':0, '10':1, 'Jack':2, 'Queen':3, 'King':4, 'Ace':5}
suits_encode = {'Clovers':0, 'Diamonds':1, 'Hearts':2, 'Spades':3}

ranks_decode = ['9', '10', 'Jack', 'Queen', 'King', 'Ace']
suits_decode = ['Clovers', 'Diamonds', 'Hearts', 'Spades']

def card_to_vector(card: Card):
  anArray = np.zeros(24)
  anArray[card_id(card)] = 1
  return anArray

def card_id(card: Card):
  return suits_encode[card.suit] * 6 + ranks_encode[card.rank]

def one_encode(cards: list) -> np.ndarray:
  anArray = np.zeros(24)
  for card in cards:
    anArray[suits_encode[card.suit] * 6 + ranks_encode[card.rank]] = 1
  return anArray

def one_decode(vector) -> Card:
  v = np.argmax(vector)
  return decode(v)

'''
  Decodes intiger value to card
'''
def decode(val: int) -> Card:
  return Card(suits_decode[val // 6], ranks_decode[val % 6])

def card_names():
    deck = []
    for suit in suits_decode:
      for rank in ranks_decode:
          deck.append(f"{rank[0]}{suit[0]}")
    return deck