from Coach import Coach
from searchgame.searchgame import SearchGame as Game
from searchgame.Nnet import NNetWrapper as nn

args = {
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
}

game = Game()
nnet = nn(game)

c = Coach(game, nnet, args)
c.learn()
