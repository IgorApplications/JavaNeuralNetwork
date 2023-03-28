package com.iapp.app;

import com.iapp.nn.DataManager;
import com.iapp.nn.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

public class NNFrame extends JFrame {

    private final NNController controller;

    public NNFrame(NNController controller) {
        super("Neural Network");
        this.controller = controller;

        setSize(500, 500);
        setResizable(false);
        setLocationRelativeTo(null);

        initialize();
    }

    public void initialize() {
        var content = new JPanel();
        content.setBorder(BorderFactory.createEmptyBorder(10,10,10,10));
        getContentPane().add(content);
        content.setLayout(new BoxLayout(content, BoxLayout.Y_AXIS));

        var description = new JLabel("<html>Select an image so that the neural network determines the number</html>");
        description.setFont(new Font("FONT", Font.PLAIN, 20));

        var button = new JButton("Load");
        button.setFont(new Font("FONT", Font.PLAIN, 20));

        var label = new JLabel("Result: ");
        label.setFont(new Font("BOLD", Font.BOLD, 20));

        content.add(description);
        content.add(button);
        content.add(label);

        var fileChooser = new JFileChooser();
        addChoiceDirectoryListener(fileChooser, JFileChooser.FILES_ONLY, label, button);
    }

    private void addChoiceDirectoryListener(JFileChooser fileChooser, int fileMode, JLabel label, JButton button) {
        button.addActionListener(e -> {
            fileChooser.setDialogTitle("Выбор файла");
            fileChooser.setFileSelectionMode(fileMode);
            int result = fileChooser.showOpenDialog(this);
            if (result == JFileChooser.APPROVE_OPTION) {
                JOptionPane.showMessageDialog(this,
                        fileChooser.getSelectedFile());

                try {

                    var path = fileChooser.getSelectedFile().getPath();
                    label.setText("Result: " + controller.prediction(path));

                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        });
    }
}
