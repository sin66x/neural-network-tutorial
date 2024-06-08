package com.example.neuralnetwork;

import java.util.Random;

public class Layer {
    private final int numberOfNeurons;
    private final int numberOfInputs;
    private final double[][] weights;
    private final double[] outputs;
    private double[] inputs;
    private final double[] biases;
    private final double[] errors;

    /**
     * Constructor to initialize the layer with specified number of neurons and inputs.
     *
     * @param numberOfNeurons Number of neurons in the layer.
     * @param numberOfInputs  Number of inputs to each neuron in the layer.
     */
    public Layer(final int numberOfNeurons, final int numberOfInputs) {
        this.numberOfNeurons = numberOfNeurons;
        this.numberOfInputs = numberOfInputs;
        this.weights = new double[numberOfNeurons][numberOfInputs];
        this.outputs = new double[numberOfNeurons];
        this.biases = new double[numberOfNeurons];
        this.errors = new double[numberOfNeurons];
        initializeWeights();
    }

    /**
     * Initialize weights and biases with random values.
     */
    private void initializeWeights() {
        final Random rand = new Random();
        for (int i = 0; i < numberOfNeurons; i++) {
            for (int j = 0; j < numberOfInputs; j++) {
                weights[i][j] = rand.nextDouble() * 2 - 1; // Random weights between -1 and 1
            }
            biases[i] = rand.nextDouble() * 2 - 1; // Random biases between -1 and 1
        }
    }

    /**
     * Perform feedforward operation for the layer.
     *
     * @param inputs The input values.
     * @return The outputs of the layer.
     */
    public double[] feedForward(final double[] inputs) {
        this.inputs = inputs;
        for (int i = 0; i < numberOfNeurons; i++) {
            double sum = 0;
            for (int j = 0; j < numberOfInputs; j++) {
                sum += inputs[j] * weights[i][j];
            }
            sum += biases[i];
            outputs[i] = sigmoid(sum); // Apply sigmoid activation function
        }
        return outputs;
    }

    /**
     * Sigmoid activation function.
     *
     * @param x Input value.
     * @return Output value after applying sigmoid activation.
     */
    private double sigmoid(final double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Calculate errors for the output layer.
     *
     * @param expected The expected output values.
     */
    public void calculateOutputLayerError(final double[] expected) {
        for (int i = 0; i < errors.length; i++) {
            errors[i] = (expected[i] - outputs[i]) * sigmoidDerivative(outputs[i]);
        }
    }

    /**
     * Calculate errors for the hidden layer.
     *
     * @param nextLayer The next layer in the neural network.
     */
    public void calculateHiddenLayerError(final Layer nextLayer) {
        for (int i = 0; i < errors.length; i++) {
            double error = 0.0;
            for (int j = 0; j < nextLayer.errors.length; j++) {
                error += nextLayer.errors[j] * nextLayer.weights[j][i];
            }
            errors[i] = error * sigmoidDerivative(outputs[i]);
        }
    }

    /**
     * Update weights and biases using error information and learning rate.
     *
     * @param learningRate The learning rate for weight updates.
     */
    public void updateWeights(final double learningRate) {
        for (int i = 0; i < numberOfNeurons; i++) {
            for (int j = 0; j < numberOfInputs; j++) {
                weights[i][j] += learningRate * errors[i] * inputs[j];
            }
            biases[i] += learningRate * errors[i];
        }
    }

    /**
     * Sigmoid derivative function.
     *
     * @param x Input value.
     * @return Derivative of the sigmoid function.
     */
    private double sigmoidDerivative(final double x) {
        return x * (1 - x);
    }
}
