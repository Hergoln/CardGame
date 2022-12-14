"""
      One Batch Man module. Contains Juliusz Malkas agents logic for
      playing undefined card game using Double Deep Q Learning 

      Makes perfect move every time.

      First 6 values are clovers, second 6 are diamonds, than hearts and than spades:
        (9 of clovers: 1000(..), 10 of clovers: 0100(..), ...)

      Input to model could be a matrix 24 x 3 + 4. 3 vectors of 0-1 encoded values of cards. Each vector
      has 24 values 1s and 0s.
          First: 1s indicate cards in the hand
          Second: 1s indicate cards discarded in whole game
          Third: 1s indicate cards discarded in current round
          Last 4 value: one-hot which suit has been played first


      Output of the model:
          softmaxed single vector of 24 values indicating which card player should play

      Training could incorporate loss function that strongly discourages using cards that model do not have in hand

      Forward pass:
          24 x 3 + 4 -> model.predict -> softmax(24x1)

  Learning:
      Each move in a game will be collected as single datapoint. 
"""

from .model import Brain
from .utils import one_encode, decode, card_names, suits_encode, card_id
from .player import OneBatchMan
from .replybuffer import ReplayBuffer

