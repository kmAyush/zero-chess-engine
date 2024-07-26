#!/usr/bin/env python
import os
import chess.pgn
from state import State

def get_XY(num_samples = None):
    X, Y = [], []
    count = 0
    for fn in os.listdir("dataset"):
        pgn = open(os.path.join("dataset", fn))
        while True:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break
            print(f"parsing game {count} got {len(X)} examples")
            count +=1
            value = {"1/2-1/2":0, "0-1":-1, "1-0":1}[game.headers["Result"]] 
            board = game.board()

            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                serialized_form = State(board).serialize()[:,:,0]
                X.append(serialized_form)
                Y.append(value)
            if num_samples is not None and len(X) > num_samples:
                return X, Y
        #break

if __name__ == "__main__":
    get_XY(1000)