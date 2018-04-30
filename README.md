# Multi-Layer-Perceptron-MLP
A Multi-Layer Perceptron MLP in Java.

-Create a new MLP with any given number of inputs, any number of outputs (can be sigmoidal or linear), and any number of hidden units (sigmoidal) in a single layer.
- Initialise the weights of the MLP to small random values
-Predict the outputs corresponding to an input vector
-Implement learning by backpropagation.


1. Train an MLP with 2 inputs, two hidden units and one output on the following examples (XOR function):
((0, 0), 0)
((0, 1), 1)
((1, 0), 1)
((1, 1), 0)

2. At the end of training, check if the MLP predicts correctly all the examples.

3. Generate 50 vectors containing 4 components each. The value of each component should be a random number between -1 and 1. These will be your input vectors. The corresponding output for each vector should be the sin() of a combination of the components. Specifically, for inputs:

[x1 x2 x3 x4]
the (single component) output should be:
sin(x1-x2+x3-x4)

4.Train an MLP on the letter recognition set available in the UCI Machine Learning repository:
http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data
The first entry of each line is the letter to be recognised (i.e. the target) and the following numbers are attributes extracted from images of the letters (i.e. your input). You can find a description of the set here:
http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
Split the dataset in a training part containing approximately 4/5 of the records, and a testing part containing the rest.
