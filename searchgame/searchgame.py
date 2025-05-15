from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
from Game import Game
from searchgame.searchgamelogic import SearchGameLogic

class SearchGame(Game):
    def __init__(self, args):
        # args.size — длина вашего массива (например, 30)
        self.size = args.size
        # всего действий = 100 числовых + len(cmds)
        self.action_size = SearchGameLogic.get_action_size()

        # obs состоит из:
        #   size элементов массива
        # + 1 target
        # + 1 cur
        # + 5 vars
        # + action_size маски допустимых ходов
        # + 1 done_flag
        # + 1 found_flag
        # + 1 cmp_count
        self.obs_dim = (
            self.size  # array
            + 1        # target
            + 1        # cur
            + 5        # vars
            + self.action_size  # mask
            + 3        # done, found, cmp_count
        )

    def getInitBoard(self):
        arr = np.random.randint(1, 101, size=self.size)
        if self.args.sorted:
            arr.sort()
        tgt = int(np.random.choice(arr))
        return SearchGameLogic(arr, tgt).get_obs()

    def getBoardSize(self):
        # neural net wrapper будет использовать board_x = obs_dim
        return (self.obs_dim,)

    def getActionSize(self):
        return SearchGameLogic.get_action_size()

    def getNextState(self, board, player, actionIndex):
        logic = SearchGameLogic.from_obs(board)
        action = self._index_to_action(actionIndex)
        obs2, reward, done = logic.step(action)
        # reward и done теперь учитываются в MCTS/Coach
        return obs2, player

    def getValidMoves(self, board, player):
        logic = SearchGameLogic.from_obs(board)
        mask = logic.allowed
        # конвертируем mask в вектор 0/1
        vec = [0] * self.getActionSize()
        for a in mask:
            idx = self._action_to_index(a)
            vec[idx] = 1
        return np.array(vec)

    def getGameEnded(self, board, player):
        from searchgame.searchgamelogic import SearchGameLogic
        logic = SearchGameLogic.from_obs(board)
        # возвращаем +1 если нашли, -1 если закончили без нахождения, 0 иначе
        if logic.done:
            return 1 if logic.found else -1
        return 0

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