# Get Chess Board

    Watching chess on YouTube:
          
          -But why not  <random_chess_move>??  
    
    The video moves on without hearing you…

    In a parallel universe:
        The video still moves on without hearing you… but you open that position on a chess engine.

## 1. Motivation and Goals
Videos on YouTube provide a lot of content for enjoying chess games played and commented on by extraordinary players. The videos also happen to be fast and provide a lot of exposure to chess but they sometimes lack in commentary regarding alternative moves - mostly rightfully so. In such cases, the ability to have an analysis board of the current position open up in a browser could help the viewer study the game more effectively and better answer “what-if” questions that they might have.

With this project, I aim to decrease the friction to study a chess game played on a video platform. This could:
1.	Shorten the feedback-loop on learning by making the analysis board (an infallible guide for almost all purposes) available quicker.
2.	Introduce a programmatic tool for capturing chess games from videos.
3.	Make watching chess videos more fun!

The project will take a screenshot of a chess game on a video platform and return a string that represents the piece placement. 

## 2. Project in Pictures

![Prediction Flow](z_markdown_jpgs/GetChessBoard-PredictionFlow.png)

## 3. Project Contributions

1. End-to-end neural network based piece
placement generation from a screenshot. 
2. Transfer learning from an object detection model ([yolov5](https://github.com/ultralytics/yolov5) via [yolo](https://pjreddie.com/media/files/papers/yolo.pdf) using PyTorch.
3. Bottom up training of a fully connected classifier (using Tensorflow)
4. Creation of chess specific (& easily extendable) platform independent GUI.

## 4. More Information

More information about the project is available in the [presentation](Project%20Presentation%20-%20Get%20Chess%20Board.pdf) and the [report](Project%20Report%20-%20Get%20Chess%20Board.pdf) (even more detail and discussion of paths forward).

Notebook 1 through 3 contain: end-to-end demonstration of the process in [Part 1 Demo](<Get Chess Board - Part 1 - Objective - Methodology and  End-to-End Demonstration.ipynb>); demonstrations of GUI in [Part 2 - Board GUI](<Get Chess Board - Part 2a - Data Acquisition and Exploration for Chessboard Detection.ipynb>) and [Part 2 - Square GUI](<Get Chess Board - Part 2b - Data Acquisition and Exploration for Piece Identification.ipynb>); and demonstration of the detection and classification objects in [Part 3 - Board Detection](<Get Chess Board - Part 3a - Chessboard Detection - Train + Predict.ipynb>) and [Part 3 - Piece Classification](<Get Chess Board - Part 3b - Piece Classification - Train + Predict.ipynb>).