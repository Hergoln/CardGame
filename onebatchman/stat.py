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