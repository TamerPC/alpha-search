import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SearchNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SearchNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, output_size)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.fc_policy(x), dim=1)
        value = torch.tanh(self.fc_value(x))
        return policy, value


class NNetWrapper:
    def __init__(self, game):
        self.game = game
        input_size = self.game.getBoardSize()[0] * 3  # Пример: массив, цель, текущий индекс
        hidden_size = 64
        output_size = self.game.getActionSize()
        self.nnet = SearchNNet(input_size, hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)

    def train(self, examples):
        self.nnet.train()
        for epoch in range(10):
            for state, pi, v in examples:
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                pi = torch.tensor(pi, dtype=torch.float32).to(self.device)
                v = torch.tensor(v, dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()
                out_pi, out_v = self.nnet(state)
                loss_pi = -torch.sum(pi * torch.log(out_pi + 1e-8))
                loss_v = torch.sum((v - out_v.view(-1)) ** 2)
                loss = loss_pi + loss_v
                loss.backward()
                self.optimizer.step()

    def predict(self, board):
        self.nnet.eval()
        board = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi, v = self.nnet(board)
        return pi.cpu().numpy()[0], v.item()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
