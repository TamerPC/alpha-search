import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization

class SearchGameNNet:
    def __init__(self, game, args):
        self.action_size = game.getActionSize()
        self.input_shape = game.getBoardSize()
        s = Input(shape=self.input_shape)
        x = Dense(args.hidden, activation='relu')(s)
        x = BatchNormalization()(x)
        x = Dense(args.hidden, activation='relu')(x)
        pi = Dense(self.action_size, activation='softmax', name='pi')(x)
        v  = Dense(1, activation='tanh',       name='v')(x)
        self.model = Model(inputs=s, outputs=[pi, v])
        self.model.compile(optimizer=Adam(learning_rate=args.lr),
                           loss={'pi': 'categorical_crossentropy', 'v': 'mean_squared_error'})