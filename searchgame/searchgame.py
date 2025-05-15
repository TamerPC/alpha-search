from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
from Game import Game
from searchgame.searchgamelogic import SearchGameLogic

class SearchGame(Game):
    def __init__(self, args):
        self.args = args
        self.size = args.size
        self.obs_dim = 1 + self.size + 1 + 5 + self.getActionSize()

    def getInitBoard(self):
        arr = np.random.randint(1, 101, size=self.size)
        if self.args.sorted:
            arr.sort()
        tgt = int(np.random.choice(arr))
        return SearchGameLogic(arr, tgt).get_obs()

    def getBoardSize(self):
        return (self.obs_dim,)

    def getActionSize(self):
        return SearchGameLogic.get_action_size()

    def getNextState(self, board, player, actionIndex):
        logic = SearchGameLogic.from_obs(board, self.size)
        action = self._index_to_action(actionIndex)
        obs, reward, done = logic.step(action)
        return obs, player  # одномерная игра

    def getValidMoves(self, board, player):
        logic = SearchGameLogic.from_obs(board, self.size)
        mask = np.zeros(self.getActionSize(), dtype=np.int8)
        for a in logic.allowed:
            mask[self._action_to_index(a)] = 1
        return mask

    def getGameEnded(self, board, player):
        logic = SearchGameLogic.from_obs(board, self.size)
        return 1 if (logic.done and logic.found) else -1 if logic.done else 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    @staticmethod
    def _action_to_index(action):
        if isinstance(action, int):
            return action - 1
        cmds = ['cmp','set','add','sub','mul','div',
                'ifless','ifless_close','ifbigger','ifbigger_close','mov','end']
        return 100 + cmds.index(action)

    @staticmethod
    def _index_to_action(index):
        if index < 100:
            return index + 1
        cmds = ['cmp','set','add','sub','mul','div',
                'ifless','ifless_close','ifbigger','ifbigger_close','mov','end']
        return cmds[index - 100]