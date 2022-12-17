import random
from card_game import Player, Card

class RandomPlayer(Player):
    """
    Makes random moves (but according to the rules)
    """
    def __init__(self, player):
        self.number = player

    def make_move(self, game_state: dict, was_previous_move_wrong: bool) -> Card:
        if not game_state["discard"]:
            return random.choice(game_state["hand"])
        else:
            options = list(filter(lambda card: card.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
            if len(options) > 0:
                return random.choice(options)
            else:
                return random.choice(game_state["hand"])

    def get_name(self):
        return f"RandomPlayer{self.number}"

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        pass

    def set_final_reward(self, points: dict):
        pass
