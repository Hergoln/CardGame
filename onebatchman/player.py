from card_game import Player, Card
from onebatchman.utils import card_to_vector
from . import decode, one_encode, Brain, suits_encode, one_decode, card_id
from pprint import pprint

import random
import numpy as np

from pprint import pprint

class OneBatchMan(Player):
  def __init__(self, player_no, learning=False) -> None:
    self.number = player_no
    self.learning = False
    self.memory = np.zeros(24)
    self.cur_state = None
    self.action = None
    self.cur_hand = None
    self.previous_state = None
    self.previous_action = None
    self.temp_reward = None
    self.mvs = 0
    self.random_mvs = 0
    self.model = Brain(alpha=5e-4, gamma=0.8,batch_size=32,input_dims=(3*24 + 4), mem_size=1_000,dnss=[64, 48],n_actions=24)

  def make_move(self, game_state: dict, was_previous_move_wrong: bool) -> Card:
    self.mvs += 1

    if was_previous_move_wrong:
      self.random_mvs += 1
      card = random.choice(game_state['hand'])
      self.action = card_id(card)
      self.remember_bad_move(self.action)
      return card

    self.previous_state = self.cur_state
    self.previous_action = self.action
    self.cur_state = self.create_state_vector(game_state)
    pred = self.model.predict(self.cur_state)[0] # returns vector of probabilities
    self.action = np.argmax(pred)

    # for this to work, all states, actions and rewards have to be nulled, so it wont go into hell by accident
    if self.previous_state:
      self.model.remember(self.previous_state, self.previous_action, self.temp_reward, self.cur_state, False)
      self.model.learn()

    return decode(self.action)

  def create_state_vector(self, game_state):
    self.cur_hand, discard, played = one_encode(game_state['hand']), one_encode(game_state['discard']), self.memory
    first_card = np.zeros(4)

    if game_state['discard']:
      first_card[suits_encode[game_state['discard'][0].suit]] = 1
    
    _state = np.vstack((self.cur_hand, discard, played))
    _state = np.append(_state.flatten(), first_card)
    return _state

  def remember_bad_move(self, action):
    # done = True, because if its a bad move we do not have a next state, 
    # thus passing True here will zero out second part of target computation
    self.model.remember(self.cur_state.copy(), action, -200, self.no_state(), True)
    self.model.learn()

  def get_name(self):
    return f"OneBatchMan{self.number}"

  """
    discarded cards so we can update appropriate vector
    and points rewarded for each player per card played
  """
  def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
    discarded = one_encode(discarded_cards.values())
    self.memory[discarded > 0] = 1
    self.temp_reward = point_deltas[self]

  def set_final_reward(self, points: dict):
    self.model.remember(self.cur_state, self.action, points[self], self.no_state(), True)
    self.model.learn()

    self.memory = np.zeros(24)
    self.previous_state = None
    self.previous_action = None
    self.temp_reward = None
    self.cur_state = None

  def save_model(self, fname):
    self.model.save_model(fname)


  def load_model(self, fname):
    self.model.load_model(fname)

  def loss_history(self):
    return self.model.history

  """
    State that is just a placeholder, used for example as next state for final state
    or for bad move
  """
  def no_state(self):
    return np.zeros(3*24 + 4)