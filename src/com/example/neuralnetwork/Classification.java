package com.example.neuralnetwork;

import java.util.Arrays;
import java.util.Random;

public class Classification {
    public static void main(String[] args) {
        // Initialize the neural network with 2 input neurons, 10 hidden neurons, and 1 output neuron
        final NeuralNetwork nn = new NeuralNetwork(2, 10, 1);

        // Define classification problem training data: [gender, height]
        final double[][] inputs = {
                {1, 190}, {1, 185}, {1, 192}, {1, 195}, {1, 187}, // Tall men
                {0, 175}, {0, 173}, {0, 178}, {0, 180}, {0, 182}, // Tall women
                {1, 160}, {1, 155}, {1, 150}, {1, 165}, {1, 158}, // Short men
                {0, 150}, {0, 155}, {0, 145}, {0, 152}, {0, 153}  // Short women
        };

        // Normalize heights to range [0, 1]
        double[][] normalizedInputs = Arrays.stream(inputs).map(
                inp -> new double[]{inp[0], normalizeHeight(inp[1])}
        ).toArray(double[][]::new);

        // Define expected outputs: 1 for tall, 0 for short
        final double[][] outputs = {
                {1},{1},{1},{1},{1}, // Tall
                {1},{1},{1},{1},{1}, // Tall
                {0},{0},{0},{0},{0}, // Short
                {0},{0},{0},{0},{0}  // Short
        };

        // Train the neural network
        nn.train(normalizedInputs, outputs, 1000000, 0.01);

        // Generate random test inputs for evaluation
        final double[][] testInputs = generateRandomTestInputs(40);

        // Evaluate and print results for test inputs
        for (double[] testCase : testInputs) {
            // Determine gender
            String gender = testCase[0] == 1 ? "A man" : "A woman";
            double height = testCase[1]; // Get height
            double[] result = nn.feedForward(new double[]{testCase[0], normalizeHeight(testCase[1])}); // Get neural network output
            String heightDescription = result[0] > 0.5 ? "tall" : "short"; // Determine height description based on output
            // Calculate certainty of prediction
            double certainty = Math.pow(result[0]-0.5,2);
            // Output result with certainty
            System.out.println(gender + " with " + (int)(height) + " cm is " + heightDescription + "\t(Certainty: " + certainty + ")");
        }
    }

    // Generate random test inputs
    public static double[][] generateRandomTestInputs(int numberOfTests) {
        double[][] testInputs = new double[numberOfTests][2];
        Random rand = new Random();

        for (int i = 0; i < numberOfTests; i++) {
            // Generate random gender: 0 for Female, 1 for Male
            double gender = rand.nextInt(2);
            // Generate random height: between 140 cm and 200 cm
            double height = 140 + (60 * rand.nextDouble());

            testInputs[i][0] = gender;
            testInputs[i][1] = height;
        }

        return testInputs;
    }

    // Normalize height to range [0, 1]
    private static double normalizeHeight(double height) {
        return (height - 140) / 60;
    }
}
