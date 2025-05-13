import numpy as np

class SearchGame:
    def __init__(self, n=10):
        self.n = n
        self.action_size = 3

    def getInitBoard(self):
        self.array = np.random.randint(0, 100, self.n)
        self.target = np.random.choice(self.array)
        self.current_index = 0
        self.done = False
        return self.getBoard()

    def getBoardSize(self):
        return (self.n,)

    def getActionSize(self):
        return self.action_size

    def getNextState(self, board, player, action):
        if self.done:
            return board, player

        if action == 0:
            self.current_index = min(self.current_index + 1, self.n - 1)
        elif action == 1:
            if self.array[self.current_index] == self.target:
                self.done = True
        elif action == 2:
            self.done = True

        return self.getBoard(), player

    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        if not self.done:
            valids = [1] * self.getActionSize()
        return np.array(valids)

    def getGameEnded(self, board, player):
        if self.done:
            if self.array[self.current_index] == self.target:
                return 1
            else:
                return -1
        return 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return str(board)

    def getBoard(self):
        return np.array([self.array, [self.target]*self.n, [self.current_index]*self.n])
