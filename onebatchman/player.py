from card_game import Player, Card
from onebatchman.utils import card_to_vector
from . import decode, one_encode, Brain, suits_encode
from pprint import pprint

import random
import numpy as np

from pprint import pprint

class OneBatchMan(Player):
  def __init__(self, player_no, learning=False, alpha=0.0005, batch_size=32) -> None:
    self.number = player_no
    self.learning = False
    self.memory = np.zeros(24)
    self.cur_state = None
    self.action = None
    self.cur_original_action = None
    self.cur_hand = None
    self.mvs = 0
    self.random_mvs = 0
    m = Brain(alpha=alpha, gamma=0.9, epsilon=1, batch_size=batch_size, mem_size=10_000)
    self.model = m
    self.wpmw = False
    self.rc = 0 # repeat counter

  def create_state(self, game_state):
    self.cur_hand, discard, played = one_encode(game_state['hand']), one_encode(game_state['discard']), self.memory
    first_card = np.zeros(4)

    if game_state['discard']:
      first_card[suits_encode[game_state['discard'][0].suit]] = 1
    
    _state = np.vstack((self.cur_hand, discard, played))
    _state = np.append(_state.flatten(), first_card)
    return _state

  def make_move(self, game_state: dict, was_previous_move_wrong: bool) -> Card:
    self.mvs += 1

    if not was_previous_move_wrong:
      self.cur_state = self.create_state(game_state)
      self.action = self.model.predict(self.cur_state)[0] # returns vector of probabilities
      self.cur_original_action = self.action.copy()
    else:
      self.remember_bad_move(self.action)
      self.rc += 1
      
      is_all_zeros = np.all((self.action == 0))
      if is_all_zeros:
        self.random_mvs += 1
        card_action = random.choice(game_state['hand'])
        self.action = card_to_vector(card_action)
      else:
        v = np.argmax(self.action)
        self.action[v] = 0

    return decode(np.argmax(self.action))

  def remember_bad_move(self, action):
    # possibly add change of next state
    self.model.remember(self.cur_state.copy(), action.copy(), -512, self.cur_state.copy(), False)
    self.model.learn()

  def get_name(self):
    return f"OneBatchMan{self.number}"


  """
    empty_discard_pile and empty_first_card are here because remember_transition
    has to create NEXT STATE (state_) and this state is a state after all players
    has played their cards and no card is on the table
  """
  def remember_transition(self, points, done):
    hand_ = self.cur_hand - self.action
    empty_discard_pile = np.zeros(24)
    played = self.memory.copy()
    empty_first_card = np.zeros(4)
    
    state_ = np.vstack((hand_, empty_discard_pile, played))
    state_ = np.append(state_.flatten(), empty_first_card)
    # if self.wpmw:
    #   points = points * 2
    # self.wpmw = False
    # In reality points are a penalty, because the goal of the game is to have THE LEAST no. of points
    self.model.remember(self.cur_state, self.action, points, state_, done)
    self.model.learn()


  """
    discarded cards so we can update appropriate vector
    and points rewarded for each player per card played
  """
  def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
    discarded = one_encode(discarded_cards.values())
    self.memory[discarded > 0] = 1
    self.remember_transition(point_deltas[self], False)


  """
    points rewarded at the end of the game
  """
  def set_final_reward(self, points: dict):
    self.remember_transition(points[self], True)
    self.memory = np.zeros(24)


  def save_model(self, fname):
    self.model.save_model(fname)


  def load_model(self, fname):
    self.model.load_model(fname)

  def loss_history(self):
    return self.model.history