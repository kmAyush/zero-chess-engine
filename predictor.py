#!/usr/bin/env python3
from state import State
from train import NeuralNet

from flask import Flask, Response, request
import base64
import chess
import chess.svg
import streamlit as st
import matplotlib.pyplot as plt
import torch
import traceback
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

def get_svg(s):
    board_str = chess.svg.board(s.board)
    board = board_str.encode("utf-8")
    board_svg = base64.b64encode(board).decode('utf-8')
    return board_svg

app = Flask(__name__)
eval = Evaluator()
s = State()

@app.route("/")
def hello():
    board_svg = get_svg(s)    
    return f'''
    <html>
    <body style="background-color:#22272d;">
        <img src="data:image/svg+xml;base64,{board_svg}" style="max-width:500px">
        <br><br>
        
        <div align="center">
            <h3 style="color:white;">FEN format input</h3>
            <form action="/human">
                <input name="move" style="font-size:20px;" type='text'></input>
                <input style="font-size:20px;" type='submit' value='Move'>
            </form><br>
        </div>
    '''
@app.route("/board.svg")
def board():
    return Response(chess.svg.board(board = s.board), mimetype='image/svg+xml')

#@app.route("/move", methods=["POST"])
def computer_move(s, eval):
    move = sorted(explore(s, eval), key = lambda x:x[0], reverse = s.board.turn)
    print("Neural net insight:")
    for i,m in enumerate(move[0:3]):
        print("Moves : ",m[1],", Probability ", m[0])
    s.board.push(move[0][1])

@app.route("/selfplay")
def selfplay():
    s = State()
    return_str = "<html><body style='background-color:#22272d;'><head>"
    while not s.board.is_game_over():
        computer_move(s, eval)
        return_str += f"<img width=400 height=400 src='data:image/svg+xml;base64,{get_svg(s)}'><br/>"
    print(s.board.result())
    return return_str
        

@app.route("/human")
def human_move():
    if not s.board.is_game_over():
        move = request.args.get("move", default="")
        if move is not None and move!="":
            try:
                s.board.push_san(move)
                print("Human Move = ",move)
                computer_move(s, eval)
            except Exception:
                traceback.print_exc()
    else:
        print("Game is Over")
    return hello()+f"{move}"
    
#computer_move()



if __name__ == "__main__":
    
    app.run(debug=True)
    # print(eval(s))
    # while not s.board.is_game_over():
    #     move_list = sorted(explore(s, eval), key = lambda x:x[0], reverse = s.board.turn)
    #     move = move_list[0]
    #     print(move)
    #     s.board.push(move[1])
    # svg = display.SVG(chess.svg.board(board = s.board))
    # file_name = "board_state.svg"
    # st.image(svg, output_format="auto")
    # print(s.board.result())
        #if s.board.turn == chess.WHITE:
