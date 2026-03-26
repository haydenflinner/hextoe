# Hex tic tac toe
Board:
Infinite hex grid

Move order:
X moves first.
After the first move (I'll call it anchor), each player gets two moves.
Concretely: XOOXXOO...

Win condition:
A player who gets 6 in a row wins.

# Tests
We should have small, simple / readable tests to ensure that what we code works correctly.

# Bot approach
Lets use Monte-Carlo Tree Search to determine the best three options for the player at any time.
There is a reference implementation in mcts-rs, but I'd like to avoid unnecessary Vec creation; lets specialize on our game in the name of simplicity and performance.

# Gui
We will need to give the player an egui gui for:
1. inputting moves, i.e. if the tool is to be used for reference while playing the game in another tab, we need to be able to move for both players, and calculate / display the best moves for the current player (since we implicitly derive the best moves for the other player in order to derive our own best moves, perhaps it's best to just show the best move for the player whose turn it is at any given time)
2. displaying what the best moves are (and perhaps why? or any other stats we can give, like win % or whatever?)

## Performance
Lets use data-oriented design and wgpu to ensure that we are achieving max performance. Though we should probably code something simple first and then we'll port to running on the gpu once we have something that works.

