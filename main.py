import random
import argparse

from card_game import CardGame, Player, Card
from onebatchman import OneBatchMan, card_names
from matplotlib import pyplot as plt
import tensorflow as tf
from pprint import pprint


player = 1
global_stats = []

class RandomPlayer(Player):
    """
    Makes random moves (but according to the rules)
    """
    def __init__(self):
        global player
        self.number = player
        player += 1

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
    
def print_scores(scores: dict):
    out = ""
    for player in scores:
        out += f"{player.get_name()} scored {scores[player]}; "
    print(out)

def compose_stats(statistics):
    out = ""
    for p in statistics:
        if statistics[p].games != 0:
            out += f"{p.get_name()}={'{'}wr:{(statistics[p].wins / statistics[p].games):.2f}, avg:{statistics[p].avg:.2f}{'}'} "
    return out

def tf_setup():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.debugging.set_log_device_placement(False)
    tf.config.set_visible_devices(gpus, 'GPU')

def parse():
    parser = argparse.ArgumentParser(description='Python CardGame for zkum classes')
    parser.add_argument('--n_games', type=int, help='Number of games to play')
    parser.add_argument('--delay', type=int, help='Delay between agents moves')
    parser.add_argument('--display', action='store_true', help='display card game in pygame')
    parser.add_argument('--save', action='store_true', help='save model after games all games are played out')
    parser.add_argument('--load', type=str, help='path to model')
    return parser.parse_args()

def save_results(cp_path):
    with open(cp_path + '_results' + '.txt', 'w') as filehandler:
        for item in global_stats:
            filehandler.write('%s\n' % compose_stats(item))

def checkpoint(cp_path, player, loss: list, r_mvs: list):
    player.save_model(cp_path + '.h5')
    save_results(cp_path)
    fig = plot_things(loss, r_mvs)
    fig.savefig(cp_path + '.png')
    plt.close(fig)

def plot_things(loss: list, r_mvs: list):
    fig, ax = plt.subplots(2, 1, figsize=(20, 14))

    ax[0].set_title("loss")
    ax[0].plot(loss)

    ax[1].set_title("random moves per game")
    ax[1].plot(r_mvs)

    fig.tight_layout()
    return fig

def define_path():
    import os
    count = len(os.listdir("obm-cp/"))
    return "obm-cp/obm" + str((count // 3) + 1)

class stat:
    def __init__(self) -> None:
        self.wins = 0
        self.avg = 0
        self.sum = 0
        self.games = 0

    def copy(self):
        other = stat()
        other.wins = self.wins
        other.avg = self.avg
        other.sum = self.sum
        other.games = self.games
        return other

def copy(stats):
    new_stats = {}
    for key in stats:
        new_stats[key] = stats[key].copy()
    return new_stats

def update_stats(scores, stats):
    winner = None
    maxi = 0
    for p in scores:
        stats[p].sum += scores[p]
        stats[p].games += 1
        if scores[p] >= maxi:
            maxi = scores[p]
            winner = p

    stats[winner].wins += 1
    global_stats.append(copy(stats))

def update_avg(stats):
    for p in stats:
        stats[p].avg = stats[p].sum / stats[p].games

def main():
    tf_setup()
    cp_path = define_path()


    r_mvs = []

    args = parse()
    delay = args.delay if args.delay else 1000
    n_games = args.n_games if args.n_games else 2
    display = args.display
    save = args.save
    load_path = args.load

    global player
    obm = OneBatchMan(player, learning=True)
    if load_path is not None:
        obm.load_model(load_path)
    
    player += 1

    pl0 = RandomPlayer()
    pl1 = RandomPlayer()
    pl2 = RandomPlayer()
    
    statistics = {obm: stat(), pl0: stat(), pl1: stat(), pl2: stat()}
    if args.delay:
        game = CardGame(obm, pl0, pl1, pl2, delay=delay, display=display, full_deck=False)
    else:
        game = CardGame(obm, pl0, pl1, pl2, display=display, full_deck=False)

    from tqdm import tqdm
    import time
    scores = None
    interval = 10
    r_mvs_cnt = 0
    mvs = 0
    progress_bar = tqdm(range(n_games), position=0)
    stats = tqdm(total=0, position=1, bar_format='{desc}')

    for cntr in progress_bar:
        try:
            scores = game.start()
            update_stats(scores, statistics)
            update_avg(statistics)
            r_mvs.append(obm.random_mvs - r_mvs_cnt)
            stats.set_description_str(compose_stats(statistics))
            r_mvs_cnt = obm.random_mvs
            if (cntr + 1) % interval == 0:
                print("\n\n")
                print(obm.last_pred)
                if save:
                    # freq = (card_names(), obm.model.actions_frequency)
                    checkpoint(cp_path, obm, obm.loss_history(), r_mvs)
                    # print(obm.loss_history())

        except Exception as e:
            print(e)
            if save:
                checkpoint(cp_path, obm, obm.loss_history(), r_mvs)
            print("Oh well")

    
    obm.model.memory.print_buffer('buffer_result.txt')

    if scores:
        update_stats(scores, statistics)
        update_avg(statistics)
        print(compose_stats(statistics))
        # freq = (card_names(), obm.model.actions_frequency)
    if save:
        checkpoint(cp_path, obm, obm.loss_history(), r_mvs)
    plt.show()

if __name__ == '__main__':
    main()
