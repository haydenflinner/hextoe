lets make the current player's turn more prominant so that it's easier to see in the gui.

lets run our bot logic on a background thread continously, constantly updating the gui, rather than for only a short time before displaying results.
I suspect this will fix the current code's tendency to miss obvious defences that are required: right now X can hvae 5 in a row and the suggestions for O are all over the place but none involve blocking the 6th spot.

Lets also add some performance benchmarks with criterion; I suspcet we'll move to a more 'flat array' style of memory mgmt/allocation to assist in running this on the gpu.

Lets also use (candle or linfa, I have no preference) to train up a machine learning model using self-play. This way we can harvest compute over time rather than just naive mcts. In the spirit of alphaGo Zero we'll pair our MCTS with a neural network. See ref/alpha-zero.pdf
