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
        # Load the model

    def __call__(self, s):
        data = s.serialize()[None] # Add batch dimension
        output = self.model(torch.tensor(data).float()) # Apply the model
        return output.data.item() # Convert output to scalar and return

MAXVAL = 10000
class ClassicEvaluator(object):
    values = { chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3.2, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}

    def __call__(self, s):
        if s.board.is_variant_win():
            if s.turn == chess.WHITE:
                return MAXVAL
            else:
                return -MAXVAL

        if s.board.is_variant_loss():
            if s.turn == chess.WHITE:
                return -MAXVAL
            else:
                return MAXVAL
        val = 0
        pm = s.board.piece_map()
        for x in pm:
            tval = self.values[pm[x].piece_type]
            if pm[x].color == chess.WHITE:
                val+= tval
            else:
                val -= tval
        return val

def computer_minimax(s, eval, a,b, depth=2,big=False):
    if depth==0 or s.board.is_game_over():
        return eval(s)
    turn = s.board.turn
    if turn==chess.WHITE:
        ret = -MAXVAL
    else:
        ret = MAXVAL

    for e in s.edges():
        s.board.push(e)
        tval = computer_minimax(s, eval, a,b, depth-1, big)
        if turn == chess.WHITE:
            ret = max(ret, tval)
            a = max(a,ret)
            if a>=b:
                break
        else:
            ret = min(ret, tval)
            b = min(b, ret)
            if a>=b:
                break
        s.board.pop()

    return ret


def explore(s,eval):
    move_probs = [] # List of (value, move) tuples
    for edge in s.edges():
        s.board.push(edge) 
        move_probs.append((computer_minimax(s,v, a=-MAXVAL, b=MAXVAL), edge)) 
        s.board.pop() 

    return move_probs

def get_svg(s):
    board_str = chess.svg.board(s.board) # Get SVG of board
    board = board_str.encode("utf-8") # Encode to bytes
    board_svg = base64.b64encode(board).decode('utf-8') # Base64 encode and decode to string
    # Why base64 encode? Because we can't directly display SVG in HTML
    # So we convert it to a format that can be displayed in HTML
    return board_svg

app = Flask(__name__) # Create a Flask app
# __name__ is a special variable in Python that is the name of the current module
# on running script, __name__ = __main__
# on importing script, __name__ = name of the script

eval = ClassicEvaluator()
s = State()

@app.route("/")
def home():
    return_str = open("index.html").read() 
    return return_str.replace('start', s.board.fen()) # Replace 'start' with current board state

@app.route("/board.svg")
def board():
    return Response(chess.svg.board(board = s.board), mimetype='image/svg+xml') # Return SVG of board

def computer_move(s, eval):
    move = sorted(explore(s, eval), key = lambda x:x[0], reverse = s.board.turn) # Sort moves by value
    print("Neural net insight:")
    for i,m in enumerate(move[0:3]):
      print("Moves : ",m[1],", Probability ", m[0])
    try:
      s.board.push(move[0][1])
    except Exception:
      return None # If no legal moves, return None

@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get('move',default="") # Get move from URL parameter
    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move) # Push move to board in SAN format
        computer_move(s, eval)
      except Exception:
        traceback.print_exc() # Print traceback if error occurs
      response = app.response_class(
        response=s.board.fen(), # Return FEN of board
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


@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    
    # Parse move
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default='')) 
    promotion = True if request.args.get('promotion', default='') == 'true' else False 

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move) # Push move to board in SAN format
        computer_move(s, eval)
      except Exception:
        traceback.print_exc()
    response = app.response_class(response=s.board.fen(), status=200) 
    return response

  print("GAME IS OVER")
  response = app.response_class(response="game over", status=200) 
  return response

@app.route("/new_game")
def newgame():
  s.board.reset()
  response = app.response_class(response=s.board.fen(), status=200)
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
        

if __name__ == "__main__":
    if os.getenv("SELFPLAY") is not None:
        s = State()
        while not s.board.is_game_over():
            computer_move(s, eval)
        print(s.board)
        print(s.board.result())
    else:
        app.run(debug=True)
        
    # svg = display.SVG(chess.svg.board(board = s.board))
    # file_name = "board_state.svg"
    # st.image(svg, output_format="auto")
    # print(s.board.result())
        #if s.board.turn == chess.WHITE:
