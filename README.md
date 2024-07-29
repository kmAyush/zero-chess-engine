Zero Knowledge Chess Engine
-----

The chess engine is unaware of the type of characters, strategy and their importance in a chess game. It learned to play chess from training on 6 million examples of chess state and result performed by the player.
<div align="center" width="100px">
  
https://github.com/user-attachments/assets/69a9e805-5809-4738-8197-7049ef254890

</div>

* The value function is about being rewarded for taking action like the winner and punish for the one who lost that game.<br>
Value - {1 for White win, 0 for draw, -1 for Black loss}
* It utilizes neural network to sort search tree.
* The search tree also considers the special moves - castling and en peasant.
* Training on larger amount of data from 10k examples to 100k then 6M examples has shown improvement yet it still makes silly moves.

About
-----
* Trained on 9 simple CNN layer combined with Relu.
* Training take serialized data of dimension 8x8x5 bit matrix obtained from PGN chess dataset taken from <a href="https://www.kaggle.com/datasets/ironicninja/raw-chess-games-pgn">kaggle</a>.<br>
  (8x8) chess squares x 4 bit represents hashed number + 1 turn bit = 257 bits needed
* Drag and drop feature.
* Highlight source square.
* Selfplay option

Usage
-----
```bash
pip install -r requirements.txt
chmod +x play_chess.py
./play_chess.py     
```
Then, go to http://localhost:5000/


Limitation - Not comparable to human in intelligence, but shown improvement as dataset size increased.

Acknowledgement
-----
* George Hotz : twitch-chess<br>
https://www.youtube.com/watch?v=RFaFmkCEGEs

* Dataset - kaggle
* UI - chessboard.js




