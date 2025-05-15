from Coach import Coach
from searchgame.searchgame import SearchGame as Game
from searchgame.network.NNet import NNetWrapper as nn

import logging
import coloredlogs

from Coach import Coach
from searchgame import SearchGame as Game
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = dotdict({
    'size': 20,
    'sorted': False,
    'numIters': 100,
    'numEps': 20,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 40,
    'cpuct': 1.0,
    'hidden': 128,
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 10,
    'checkpoint': './checkpoint',
    'load_model': False,
    'load_folder_file': ('./checkpoint','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...',
                 args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(*args.load_folder_file)
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == '__main__':
    main()

