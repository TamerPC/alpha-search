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
        arr = np.random.randint(1,101,size=self.args.size)
        if self.args.sorted:
            arr.sort()
        tgt = int(np.random.choice(arr))
        self.logic = SearchGameLogic(arr,tgt)
        return self.logic.reset()

    def getBoardSize(self):
        return (self.obs_dim,)

    def getActionSize(self):
        """
        Возвращает общее число возможных действий (числовые + строковые команды).
        """
        cmds = ['cmp','set','add','sub','mul','div',
                'ifless','ifless_close','ifbigger','ifbigger_close','mov','end']
        return 100 + len(cmds)

    def getNextState(self, board, player, actionIndex):
        # Преобразуем индекс действия в саму команду
        action = self._index_to_action(actionIndex)
        obs, reward, done = self.logic.step(action)
        # В однопользовательской игре игрок не меняется
        return obs, player

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
    
    @staticmethod
    def _index_to_action(index):
        """
        Обратное преобразование индекса в действие.
        Для индексов 0-99 возвращает число (1-100),
        для остальных — строковую команду.
        """
        if index < 100:
            return index + 1
        cmds = ['cmp','set','add','sub','mul','div',
                'ifless','ifless_close','ifbigger','ifbigger_close','mov','end']
        return cmds[index - 100]