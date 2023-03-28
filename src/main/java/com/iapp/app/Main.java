package com.iapp.app;

public class Main {

    // training = 60_000
    // testing = 10_000

    public static void main(String[] args) {
        var controller = new NNController();
        var frame = new NNFrame(controller);
        frame.setVisible(true);
    }

}
