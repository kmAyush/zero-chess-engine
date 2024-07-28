#!/usr/bin/env python3
from state import State
from train import NeuralNet
import chess
import torch
import warnings
warnings.filterwarnings("ignore")

class Evaluator:
    def __init__(self):
        self.model = torch.load('models/examples_10k.pth')

    def __call__(self, s):
        data = s.serialize()[None]
        output = self.model(torch.tensor(data).float())
        return output.data.item()

def explore(s,eval):
    move_probs = []
    for edge in s.edges():
        s.board.push(edge)
        move_probs.append((eval(s), edge))
        s.board.pop()

    return move_probs


if __name__ == "__main__":
    eval = Evaluator()
    s = State()
    print(eval(s))
    while not s.board.is_game_over():
        move_list = sorted(explore(s, eval), key = lambda x:x[0], reverse = s.board.turn)
        move = move_list[0]
        print(move)
        s.board.push(move[1])
    print(s.board.result())
        #if s.board.turn == chess.WHITE:
