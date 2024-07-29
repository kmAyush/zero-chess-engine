#!/usr/bin/env python3
from state import State
from train import NeuralNet

from flask import Flask, Response, request
import os
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
        self.model = torch.load('models/examples_6M.pth', map_location="cpu")

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
def home():
    return_str = open("index.html").read()
    return return_str.replace('start', s.board.fen())

@app.route("/board.svg")
def board():
    return Response(chess.svg.board(board = s.board), mimetype='image/svg+xml')

#@app.route("/move", methods=["POST"])
def computer_move(s, eval):
    move = sorted(explore(s, eval), key = lambda x:x[0], reverse = s.board.turn)
    print("Neural net insight:")
    for i,m in enumerate(move[0:3]):
      print("Moves : ",m[1],", Probability ", m[0])
    try:
      s.board.push(move[0][1])
    except Exception:
      return None

@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get('move',default="")
    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, eval)
      except Exception:
        traceback.print_exc()
      response = app.response_class(
        response=s.board.fen(),
        status=200
      )
      return response
  else:
    print("GAME IS OVER")
    response = app.response_class(
      response="game over",
      status=200
    )
    return response
  return home()

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, eval)
      except Exception:
        traceback.print_exc()
    response = app.response_class(
      response=s.board.fen(),
      status=200
    )
    return response

  print("GAME IS OVER")
  response = app.response_class(
    response="game over",
    status=200
  )
  return response

@app.route("/new_game")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response


if __name__ == "__main__":
  if os.getenv("SELFPLAY") is not None:
    s = State()
    while not s.board.is_game_over():
      computer_move(s, eval)
      print(s.board)
    print(s.board.result())
  else:
    app.run(debug=True)



@app.route("/selfplay")
def selfplay():
    s = State()
    return_str = '''<html><body style="background-color:#22272d;margin:10px;"><div align="center"
    '''
    while not s.board.is_game_over():
        computer_move(s, eval)
        return_str += f"""
        <img width=400 height=400 src='data:image/svg+xml;base64,{get_svg(s)}'>
        """
    return_str+='''
    </div>
    '''
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
            response = app.response_class(
                response = s.board.fen(),
                status = 200
            )
    else:
        print("Game is Over")
    return home()
    
#computer_move()



if __name__ == "__main__":
    if os.getenv("SELFPLAY") is not None:
        s = State()
        while not s.board.is_game_over():
            computer_move(s, eval)
        print(s.board)
        print(s.board.result())
    else:
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
