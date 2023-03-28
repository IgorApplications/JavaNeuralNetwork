package com.iapp.app;

import com.iapp.nn.DataManager;
import com.iapp.nn.NeuralNetwork;

import java.io.*;
import java.util.Map;

public class NNController {

    private final NeuralNetwork nn;
    private final DataManager dataManager;

    public NNController() {
        nn = new NeuralNetwork();
        dataManager = new DataManager();
    }

    // dangerous!
    public void restudy() throws IOException {
        var result = nn.study(
                dataManager.readImagesByLabels(1_000, DataManager.MNIST_TRAINING),
                dataManager.readImagesByLabels(1_000, DataManager.MNIST_TESTING));

        var writer1 = new BufferedOutputStream(new FileOutputStream("weights01.txt", false));
        writer1.write(serialize(result.getKey()));
        writer1.close();

        var writer2 = new BufferedOutputStream(new FileOutputStream("weights12.txt", false));
        writer2.write(serialize(result.getValue()));
        writer2.close();
    }

    public int prediction(String absPath) throws IOException {
        var entry = readWeights();
        var image = dataManager.readImage(absPath);
        return nn.prediction(image, entry.getKey(), entry.getValue());
    }

    private Map.Entry<double[][], double[][]> readWeights() throws IOException {
        var reader1 = new BufferedInputStream(new FileInputStream("weights01.txt"));
        double[][] weights01 = deserialize(reader1.readAllBytes());
        reader1.close();

        var reader2 = new BufferedInputStream(new FileInputStream("weights12.txt"));
        double[][] weights12 = deserialize(reader2.readAllBytes());
        reader2.close();

        return new Map.Entry<>() {
            @Override
            public double[][] getKey() {
                return weights01;
            }

            @Override
            public double[][] getValue() {
                return weights12;
            }

            @Override
            public double[][] setValue(double[][] value) {
                throw new IllegalArgumentException();
            }
        };
    }

    private byte[] serialize(double[][] obj)  {
        try (ByteArrayOutputStream byteArray = new ByteArrayOutputStream();
             ObjectOutputStream byteWriter = new ObjectOutputStream(byteArray)) {
            byteWriter.writeObject(obj);
            return byteArray.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private double[][] deserialize(byte[] bytes) {
        try (ByteArrayInputStream byteArray = new ByteArrayInputStream(bytes);
             ObjectInputStream bytesReader = new ObjectInputStream(byteArray)) {
            return (double[][]) bytesReader.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
