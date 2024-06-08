package com.example.neuralnetwork;

public class NeuralNetwork {
    private final Layer hiddenLayer;
    private final Layer outputLayer;

    /**
     * Constructor to initialize the neural network with specified neurons
     * for the input, hidden, and output layers.
     *
     * @param inputNeurons   Number of neurons in the input layer.
     * @param hiddenNeurons  Number of neurons in the hidden layer.
     * @param outputNeurons  Number of neurons in the output layer.
     */
    public NeuralNetwork(final int inputNeurons, final int hiddenNeurons, final int outputNeurons) {
        hiddenLayer = new Layer(hiddenNeurons, inputNeurons); // Hidden layer initialized
        outputLayer = new Layer(outputNeurons, hiddenNeurons); // Output layer initialized
    }

    /**
     * Feedforward method to compute the outputs of the network given the inputs.
     *
     * @param inputs The input values.
     * @return The output values from the output layer.
     */
    public double[] feedForward(final double[] inputs) {
        final double[] hiddenOutputs = hiddenLayer.feedForward(inputs); // Feedforward through hidden layer
        return outputLayer.feedForward(hiddenOutputs); // Feedforward through output layer
    }

    /**
     * Training method to train the neural network using backpropagation.
     *
     * @param trainingInputs  The input data for training.
     * @param trainingOutputs The expected output data for training.
     * @param epochs          Number of epochs to train the network.
     * @param learningRate    The learning rate for weight updates.
     */
    public void train(final double[][] trainingInputs, final double[][] trainingOutputs, final int epochs, final double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) { // Loop through each epoch
            for (int i = 0; i < trainingInputs.length; i++) { // Loop through each training sample
                // Feedforward through the network
                feedForward(trainingInputs[i]);

                // Calculate error and backpropagate
                outputLayer.calculateOutputLayerError(trainingOutputs[i]);
                hiddenLayer.calculateHiddenLayerError(outputLayer);

                // Update weights and biases based on calculated errors
                outputLayer.updateWeights(learningRate);
                hiddenLayer.updateWeights(learningRate);
            }
        }
    }

    public static void main(String[] args) {
        xor();
        and();
    }

    public static void xor() {
        final NeuralNetwork nn = new NeuralNetwork(2, 20, 1); // Initialize the neural network

        // XOR problem training data
        final double[][] inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        final double[][] outputs = { {0}, {1}, {1}, {0} };

        nn.train(inputs, outputs, 1000000, 0.1); // Train the neural network

        // Test the trained neural network
        for (final double[] input : inputs) {
            final double[] result = nn.feedForward(input);
            System.out.println(input[0] + " XOR " + input[1] + " = " + result[0]);
        }
    }

    public static void and() {
        final NeuralNetwork nn = new NeuralNetwork(2, 20, 1); // Initialize the neural network

        // AND problem training data
        final double[][] inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        final double[][] outputs = { {0}, {1}, {1}, {1} };

        nn.train(inputs, outputs, 1000000, 0.1); // Train the neural network

        // Test the trained neural network
        for (final double[] input : inputs) {
            final double[] result = nn.feedForward(input);
            System.out.println(input[0] + " AND " + input[1] + " = " + result[0]);
        }
    }
}
