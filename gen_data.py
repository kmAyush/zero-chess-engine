#!/usr/bin/env python
import os
import chess.pgn
import numpy as np
from state import State

def get_XY(num_samples = None):
    X, Y = [], []
    count = 0
    countX = len(X)
    values = {"1/2-1/2":0, "0-1":-1, "1-0":1}
    if num_samples is not None:
        print(num_samples)
    for fn in os.listdir("dataset"):
        pgn = open(os.path.join("dataset", fn))
        while True:

            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            print(f"parsing game {count} got {len(X)} examples")
            result = game.headers["Result"]
            if result not in values:
                continue
            value = values[result]
            board = game.board()

            for i, move in enumerate(game.mainline_moves()):   
                board.push(move)
                serialized_form = State(board).serialize()
                X.append(serialized_form)
                Y.append(value)
            if num_samples is not None and len(X) > num_samples:
                return X, Y
            
            count += 1
        #break

if __name__ == "__main__":
    X, Y = get_XY(6e6)
    if not os.path.exists("serialized_data"):
        os.makedirs("serialized_data")
    np.savez("serialized_data/dataset_6M.npz", X, Y)
