from __future__ import print_function
import sys
sys.path.append('..')

import numpy as np
from Game import Game
from searchgame.searchgamelogic import SearchGameLogic

class SearchGame(Game):
    def __init__(self, args):
        self.args = args
        self.obs_dim = 12 + 5 + 100

    def getInitBoard(self):
        array = np.random.randint(1, 100, size=self.args.size)
        if self.args.sorted:
            array.sort()
        target = np.random.choice(array)
        self.logic = SearchGameLogic(array, target)
        obs = self.logic.reset()
        return obs

    def getBoardSize(self):
        return (self.obs_dim,)

    def getActionSize(self):
        return 100 + 11  # 1–100 чисел + 11 строковых команд

    def getNextState(self, board, player, action):
        obs, reward, done = self.logic.step(action)
        return obs, done

    def getValidMoves(self, board, player):
        mask = np.zeros(self.getActionSize(), dtype=np.int8)
        for a in self.logic.allowed:
            idx = self._action_to_index(a)
            mask[idx] = 1
        return mask

    def getGameEnded(self, board, player):
        return 1 if (self.logic.done and self.logic.found) else -1 if self.logic.done else 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    def _action_to_index(self, action):
        # маппинг action->индекс в векторе
        if isinstance(action, int): return action-1
        cmd_list = ['cmp','set','add','sub','mul','div',
                    'ifless','ifless_close','ifbigger','ifbigger_close','mov','end']
        return 100 + cmd_list.index(action)