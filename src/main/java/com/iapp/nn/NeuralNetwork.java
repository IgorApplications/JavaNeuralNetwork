package com.iapp.nn;

import java.util.Map;
import java.util.Random;

/**
 * Data Science
 * @author Igor Ivanov
 * @version 1.0
 * */
public class NeuralNetwork {

    private static final Random random = new Random();

    public NeuralNetwork() {}

    public Map.Entry<double[][], double[][]> study(Map.Entry<int[][], int[]> train, Map.Entry<int[][], int[]> test) {
        var images = divide(toDoubleMatrix(train.getKey()), 255);
        var oneHotLabels = new double[train.getValue().length][10];
        for (int i = 0; i < train.getValue().length; i++) {
            oneHotLabels[i][train.getValue()[i]] = 1;
        }
        var labels = oneHotLabels;

        var testImages = divide(toDoubleMatrix(test.getKey()), 255);
        var testLabels = new double[test.getValue().length][10];
        for (int i = 0; i < test.getValue().length; i++) {
            testLabels[i][test.getValue()[i]] = 1;
        }

        double alpha = 0.005f;
        int iterations = 500, hiddenSize = 200;
        int pixelsPerImage = 784, numLabels = 10;
        int batchSize = 10;

        var weights01 = random(0.02, pixelsPerImage, hiddenSize, 0.01);
        var weights12 = random(0.2, hiddenSize, numLabels, 0.1);

        for (int iteration = 0; iteration < iterations; iteration++) {
            float error = 0.0f;
            int correctCnt = 0;

            for (int j = 0; j < images.length / batchSize; j++) {

                int batchStart = j * batchSize;
                int batchEnd = (j + 1) * batchSize;

                var layer0 = sub(images, batchStart, batchEnd);
                var layer1 = tanh(dot(layer0, weights01));

                var dropoutMask = generateDropoutMask(layer1.length, layer1[0].length);
                layer1 = multiply(multiply(layer1, dropoutMask), 2);
                var layer2 = softmax(dot(layer1, weights12));

                error += sumAll(pow(subtract(sub(labels, batchStart, batchEnd), layer2), 2));
                for (int k = 0; k < batchSize; k++) {

                    if (argmax(sub(layer2, k, k + 1))
                            == argmax(sub(labels, batchStart + k, batchEnd + k))) {
                        correctCnt++;
                    }

                }

                var layer2Delta = divide(subtract(sub(labels, batchStart, batchEnd), layer2), batchSize);
                var layer1Delta = multiply(dot(layer2Delta, transpose(weights12)), derivativeOfTanh(layer1));
                layer1Delta = multiply(layer1Delta, dropoutMask);

                weights12 = add(weights12, multiply(dot(transpose(layer1), layer2Delta), alpha));
                weights01 = add(weights01, multiply(dot(transpose(layer0), layer1Delta), alpha));
            }

            if (iteration % 10 == 0) {
                float testError = 0.0f;
                float testCorrectCnt = 0;

                for (int i = 0; i < testImages.length; i++) {
                    var layer0 = sub(testImages, i, i + 1);
                    var layer1 = tanh(dot(layer0, weights01));
                    var layer2 = dot(layer1, weights12);

                    testError += sumAll(pow(subtract(sub(testLabels, i, i + 1), layer2), 2));
                    if (argmax(layer2) == argmax(sub(testLabels, i, i + 1))) testCorrectCnt++;
                }

                System.out.printf("I = %d, Test-Err=%f, Test-Acc=%f, Train-Err=%f, Train-Acc=%d%n",
                        iteration, testError / testImages.length, testCorrectCnt / testImages.length,
                        error / images.length, correctCnt / images.length);
            }
        }

        double[][] finalWeights0 = weights01;
        double[][] finalWeights1 = weights12;

        return new Map.Entry<>() {
            @Override
            public double[][] getKey() {
                return finalWeights0;
            }

            @Override
            public double[][] getValue() {
                return finalWeights1;
            }

            @Override
            public double[][] setValue(double[][] value) {
                throw new UnsupportedOperationException();
            }
        };
    }

    public int prediction(int[] image, double[][] weights01, double[][] weights12) {
        int[][] imageMatrix = new int[1][image.length];
        imageMatrix[0] = image;

        var layer0 = divide(toDoubleMatrix(imageMatrix), 255);
        var layer1 = tanh(dot(layer0, weights01));
        var layer2 = dot(layer1, weights12);

        return argmax(layer2);
    }

    private double[][] dot(double[][] first, double[][] second) {
        double[][] result = new double[first.length][second[0].length];

        for (int i = 0; i < first.length; i++) {
            var clonedSecond = clone(second);

            for (int k = 0; k < second.length; k++) {
                for (int m = 0; m < second[k].length; m++) {
                    clonedSecond[k][m] *= first[i][k];
                }
            }

            var arr = new double[clonedSecond[0].length];
            for (int j = 0; j < clonedSecond[0].length; j++) {
                double sum = 0;

                for (double[] doubles : clonedSecond) {
                    sum += doubles[j];
                }

                arr[j] = sum;
            }
            result[i] = arr;
        }

        return result;
    }

    // activation methods -------------------
    /** tanh(X) = sin(X) / cos(X) */
    private double[][] tanh(double[][] data) {
        data = clone(data);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = Math.tanh(data[i][j]);
            }
        }
        return data;
    }

    /**
     * derivative tanh^2(X) = 1 / cos^2(X)
     * 1 - tanh^2(X) = 1 / cos^2(X)
     */
    private double[][] derivativeOfTanh(double[][] data) {
        data = clone(data);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (1 - Math.pow(data[i][j], 2));
            }
        }
        return data;
    }

    /** e^x / sum [][] -> [] */
    private double[][] softmax(double[][] data) {
        var temp = clone(data);
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[i].length; j++) {
                temp[i][j] = Math.exp(data[i][j]);
            }
        }

        var sum = sum(temp);
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[i].length; j++) {
                temp[i][j] = temp[i][j] / sum[i];
            }
        }

        return temp;
    }

    // util methods ------------------------------------------
    /**
     * cuts vectors from a matrix
     * */
    private double[][] sub(double[][] matrix, int start, int end) {
        if (end > matrix.length) end = matrix.length;
        var newMatrix = new double[end - start][];

        int index = 0;
        for (int i = start; i < end; i++) {
            newMatrix[index++] = matrix[i].clone();
        }

        return newMatrix;
    }

    /**
     * converts integer numbers to double numbers
     * */
    private double[][] toDoubleMatrix(int[][] matrix) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[i][j] = matrix[i][j];
            }
        }

        return newMatrix;
    }

    /**
     * flips the matrix, swaps dimensions
     * */
    private double[][] transpose(double[][] matrix) {
        var newMatrix = new double[matrix[0].length][matrix.length];

        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                newMatrix[i][j] = matrix[j][i];
            }
        }

        return newMatrix;
    }

    /**
     * searches for the maximum element of a matrix and
     * returns the index of its representation as a vector
     * */
    private int argmax(double[][] matrix) {
        double max = matrix[0][0];
        int index = 0;

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (max < matrix[i][j]) {
                    max = matrix[i][j];
                    index = i * matrix[0].length + j;
                }
            }
        }

        return index;
    }

    /**
     * adds all the elements of the matrix
     * */
    private double sumAll(double[][] matrix) {
        double sum = 0;

        for (var doubles : matrix) {
            for (var num : doubles) {
                sum += num;
            }
        }

        return sum;
    }

    /**
     * raises each element of the matrix to the specified power
     * */
    private double[][] pow(double[][] matrix, int degree) {
        var newMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[i][j] = Math.pow(matrix[i][j], degree);
            }
        }

        return newMatrix;
    }

    /**
     * multiplies two matrices
     * */
    private double[][] multiply(double[][] first, double[][] second) {
        var newMatrix = new double[first.length][first[0].length];

        for (int i = 0; i < first.length; i++) {
            for (int j = 0; j < first[i].length; j++) {
                newMatrix[i][j] = first[i][j] * second[i][j];
            }
        }

        return newMatrix;
    }

    /**
     * multiplies each element of the matrix by the specified value
     * */
    private double[][] multiply(double[][] first, double num) {
        var newMatrix = new double[first.length][first[0].length];

        for (int i = 0; i < first.length; i++) {
            for (int j = 0; j < first[i].length; j++) {
                newMatrix[i][j] = first[i][j] * num;
            }
        }

        return newMatrix;
    }

    /**
     * adds two matrices
     * */
    private double[][] add(double[][] first, double[][] second) {
        var newMatrix = new double[first.length][first[0].length];

        for (int i = 0; i < first.length; i++) {
            for (int j = 0; j < first[i].length; j++) {
                newMatrix[i][j] = first[i][j] + second[i][j];
            }
        }

        return newMatrix;
    }

    /**
     * subtracts the given value from each element of the matrix
     * */
    private double[][] subtract(double[][] matrix, double deductible) {
       var newMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[i][j] = matrix[i][j] - deductible;
            }
        }

        return newMatrix;
    }

    /**
     * subtracts the first matrix from the second
     * */
    private double[][] subtract(double[][] first, double[][] second) {
        var newMatrix = new double[first.length][first[0].length];

        for (int i = 0; i < first.length; i++) {
            for (int j = 0; j < first[i].length; j++) {
                newMatrix[i][j] = first[i][j] - second[i][j];
            }
        }

        return newMatrix;
    }

    /**
     * divides each matrix element by a parameter
     * */
    private double[][] divide(double[][] matrix, double dividend) {
        var newMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[i][j] = matrix[i][j] /  dividend;
            }
        }

        return newMatrix;
    }

    /**
     * adds the vectors of each matrix and returns a vector
     * */
    private double[] sum(double[][] matrix) {
        var sum = new double[matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum[i] += matrix[i][j];
            }
        }

        return sum;
    }

    // ------------------------------------------

    /**
     * Generates random values using the following formula:
     * coef * random - sub
     * */
    private double[][] random(double coef, int width, int height, double sub) {
        var matrix = new double[width][height];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = coef * random.nextDouble() - sub;
            }
        }

        return matrix;
    }

    /**
     * generates a mask that nulls out random values by about 50%
     * */
    private double[][] generateDropoutMask(int width, int height) {
        var mask = new double[width][height];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                mask[i][j] = random.nextInt(2);
            }
        }

        return mask;
    }

    /**
     * creates new links with the same values
     * */
    private double[][] clone(double[][] data) {
        var cloned = new double[data.length][];

        for (int i = 0; i < data.length; i++) {
            cloned[i] = data[i].clone();
        }

        return cloned;
    }
}
