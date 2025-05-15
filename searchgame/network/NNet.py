import numpy as np
from NeuralNet import NeuralNet
from searchgameNNet import SearchGameNNet
import argparse
import os
import shutil
import time
import random
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        super().__init__(game)
        self.nnet = SearchGameNNet(game, args)
        self.board_x, = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        boards, pis, vs = list(zip(*examples))
        B = np.array(boards)
        P = np.array(pis)
        V = np.array(vs)
        self.nnet.model.fit(B, {'pi': P, 'v': V},
                           batch_size=self.args.batch_size,
                           epochs=self.args.epochs)

    def predict(self, board):
        b = board.reshape(1, self.board_x)
        pi, v = self.nnet.model.predict(b)
        return pi[0], v[0][0]

    def save_checkpoint(self, folder, filename):
        self.nnet.model.save_weights(f"{folder}/{filename}.h5")

    def load_checkpoint(self, folder, filename):
        self.nnet.model.load_weights(f"{folder}/{filename}.h5")

# Coach.py
from Coach import Coach
from searchgame import SearchGame
from NNet import NNetWrapper
from utils import dotdict

if __name__ == "__main__":
    args = dotdict({
        'size': 20,
        'sorted': False,
        'hidden': 128,
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'numMCTSSims': 50,
        'numIters': 100,
        'numEps': 20,
        'cpuct': 1.0,
        'checkpoint': './checkpoint',
    })
    game = SearchGame(args)
    nnet = NNetWrapper(game, args)
    c = Coach(game, nnet, args)
    c.learn()