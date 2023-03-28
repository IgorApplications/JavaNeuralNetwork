package com.iapp.nn;

import javax.imageio.ImageIO;
import javax.imageio.stream.MemoryCacheImageInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Data Analytics
 * @author Igor Ivanov
 * @version 1.0
 * */
public class DataManager {

    public static final String MNIST_TRAINING = "mnist/training";
    public static final String MNIST_TESTING = "mnist/testing";
    private static final ThreadLocalRandom RANDOM = ThreadLocalRandom.current();
    private static final String[] PATHS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    public Map.Entry<int[][], int[]> readImagesByLabels(int size, String path) throws IOException {
        var imagesByLabel = size / 10;
        int[] labels = new int[imagesByLabel * 10];
        int[][] images = new int[imagesByLabel * 10][784];
        int[] imagesPosition = new int[10];

        for (int i = 0; i < labels.length; i++) {
            labels[i] = (byte) RANDOM.nextInt(10);
        }

        for (int i = 0; i < labels.length; i++) {
            var folder = new File(getClass().getResource(path + "/" + PATHS[labels[i]]).getFile());
            var children = folder.listFiles();

            var fileInputStream = new FileInputStream(children[imagesPosition[labels[i]]]);
            imagesPosition[labels[i]]++;

            var memoryCache = new MemoryCacheImageInputStream(fileInputStream);
            var image = ImageIO.read(memoryCache);

            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int[] pixel = image.getRaster().getPixel(x, y, new int[3]);
                    images[i][y * x + x] = pixel[0];
                }
            }
        }

        return new Map.Entry<>() {
            @Override
            public int[][] getKey() {
                return images;
            }

            @Override
            public int[] getValue() {
                return labels;
            }

            @Override
            public int[] setValue(int[] value) {
                throw new UnsupportedOperationException();
            }
        };
    }

    // absolute path
    public int[] readImage(String path) throws IOException {
        int[] image = new int[784];

        var file = new File(path);
        var fileInputStream = new FileInputStream(file);

        var memoryCache = new MemoryCacheImageInputStream(fileInputStream);
        var bufferedImage = ImageIO.read(memoryCache);

        long sum = 0;
        for (int y = 0; y < bufferedImage.getHeight(); y++) {
            for (int x = 0; x < bufferedImage.getWidth(); x++) {
                int[] pixel = bufferedImage.getRaster().getPixel(x, y, new int[3]);
                image[y * x + x] = pixel[0];
                sum += image[y * x + x];
            }
        }

        // convert to dark from light if required and remove noise
        long medium = sum / 784;
        System.out.println(medium);
        if (medium > 128) {
            for (int i = 0; i < image.length; i++) {
                if (image[i] < medium) {
                    image[i] = 255 - image[i];
                } else {
                    image[i] = 0;
                }
            }
        } else {
            for (int i = 0; i < image.length; i++) {
                if (image[i] < medium) {
                    image[i] = 0;
                }
            }
        }


        return image;
    }
}
