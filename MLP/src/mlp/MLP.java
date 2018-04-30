package mlp;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

/**
 *
 * @author Ben Zhesi Ning
 */
public class MLP {

    private final int numInput;

    private final int numHidden;
    private final int numOutput;
    private final double[][] WH;
    private final double[][] WO;
    private final double[][] dWH;
    private final double[][] dWO;

    private final double[] AH;
    private final double[] AO;
    private final double[] OH;
    private final double[] OO;

    private final double[] bH;
    private final double[] bO;
    private double[] I;
    boolean _tanh;

    public MLP(int ni, int nh, int no, boolean tanh) {

        numInput = ni;
        numHidden = nh;
        numOutput = no;

        WH = new double[numHidden][numInput];
        dWH = new double[numHidden][numInput];

        WO = new double[numOutput][numHidden];
        dWO = new double[numOutput][numHidden];

        I = new double[numInput];
        AH = new double[numHidden];
        OH = new double[numHidden];
        AO = new double[numOutput];
        OO = new double[numOutput];

        bH = new double[numHidden];
        bO = new double[numOutput];
        _tanh = tanh;

    }

    public void randomise() {
        for (int j = 0; j < numHidden; j++) {
            for (int i = 0; i < numInput; i++) {
                WH[j][i] = Math.random() * 0.005;
                dWH[j][i] = 0.0;
            }
            bH[j] = Math.random();
        }
        for (int j = 0; j < numOutput; j++) {
            for (int i = 0; i < numHidden; i++) {
                WO[j][i] = Math.random() * 0.005;
                dWO[j][i] = 0.0;
            }
            bO[j] = Math.random();
        }

    }

    public double[] forward(double[] input) {
        I = Arrays.copyOf(input, input.length);
        for (int j = 0; j < numHidden; j++) {
            double weight = 0.0;
            for (int i = 0; i < numInput; i++) {
                weight += WH[j][i] * I[i];
            }
            weight += bH[j];
            AH[j] = weight;
            OH[j] = f(weight);
        }
        for (int j = 0; j < numOutput; j++) {
            double weight = 0.0;
            for (int i = 0; i < numHidden; i++) {
                weight += WO[j][i] * OH[i];
            }
            weight += bO[j];
            AO[j] = weight;
            OO[j] = f(weight);
        }
        return OO;
    }

    public double backward(double[] output) {
        double[] deltaO = new double[numOutput];
        for (int j = 0; j < numOutput; j++) {
            deltaO[j] = output[j] - OO[j];
            deltaO[j] *= fdx(AO[j]);
            for (int i = 0; i < numHidden; i++) {
                dWO[j][i] += deltaO[j] * OH[i];
            }
        }

        double[] deltaH = new double[numHidden];
        for (int j = 0; j < numHidden; j++) {
            for (int k = 0; k < numOutput; k++) {
                deltaH[j] += deltaO[k] * WO[k][j];
            }
            deltaH[j] *= fdx(AH[j]);

            for (int i = 0; i < numInput; i++) {
                dWH[j][i] += deltaH[j] * I[i];
            }
        }

        double error = 0.0;
        for (int j = 0; j < numOutput; j++) {
            error += Math.pow(output[j] - OO[j], 2);
        }
        return error * 0.5;
    }

    public void updateWeights(double learningRate) {
        for (int j = 0; j < numHidden; j++) {
            for (int i = 0; i < numInput; i++) {
                WH[j][i] += learningRate * dWH[j][i];
                dWH[j][i] = 0.0;
            }
        }
        for (int j = 0; j < numOutput; j++) {
            for (int i = 0; i < numHidden; i++) {
                WO[j][i] += learningRate * dWO[j][i];
                dWO[j][i] = 0.0;
            }
        }
    }

    private double f(double x) {

        if (_tanh == true) {
            return Math.tanh(x);
        } else {
            return 1.0 / (1.0 + Math.exp(-x));
        }
    }

    private double fdx(double x) {

        if (_tanh == true) {
            return 1.0 - Math.pow(f(x), 2);
        } else {
            return f(x) * (1.0 - f(x));
        }

    }

    public static void main(String[] args) throws FileNotFoundException {

        int input = -1;
        Scanner reader = new Scanner(System.in);
        System.out.println("1 for xor, 2 for vetor, 3 for letter\n");
        while (input != 1 && input != 2 && input != 3) {
            System.out.println("Enter a number: \n");
            input = reader.nextInt();
        }
        switch (input) {
            case 1:
                MLP xorNet = new MLP(2, 2, 1, false);
                xorNet.xor(xorNet, 0.55, 0.33, 100000);
                break;
            case 2:
                MLP vectorNet = new MLP(4, 13, 1, true);
                vectorNet.vectors(vectorNet, 0.0197529, 0.095, 100000);
                break;
            case 3:
                MLP letterNet = new MLP(16, 12, 26, false);
                letterNet.letter(letterNet, 0.076997529, 0.895, 1000);
                break;
            default:
                break;
        }

    }

    public void letter(MLP object, double learningRate, double percentageUpdate, int epochs) throws FileNotFoundException {
        File file = new File("letter-recognition.data");

        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();

        Scanner sc = new Scanner(file);

        while (sc.hasNextLine()) {
            String[] line = sc.nextLine().split(",");

            double[] output = new double[26];
            output[line[0].charAt(0) - 'A'] = 1.0;

            double[] input = new double[line.length - 1];
            for (int i = 1; i < input.length; i++) {
                input[i - 1] = Double.parseDouble(line[i]);
            }
            inputs.add(input);
            outputs.add(output);
        }
        sc.close();

        int maxEpochs = epochs;
        double error = 0.0;
        MLP mlp = object;
        mlp.randomise();
        for (int e = 0; e < maxEpochs; e++) {
            error = 0.0;
            for (int p = 0; p < inputs.size() * 0.75; p++) {

                mlp.forward(inputs.get(p));
                error += mlp.backward(outputs.get(p));
                if (Math.random() < percentageUpdate) {
                    mlp.updateWeights(learningRate);
                }
            }
            System.out.println(String.format("Error at %d is %.12f", e, error));
        }

        double perdictedError = 0.0;
        int count = (int) (inputs.size() * 0.75);
        int success = 0;
        for (int t = count; t < inputs.size(); t++) {
            double[] predicted = mlp.forward(inputs.get(t));

            double max = Double.NEGATIVE_INFINITY;
            char expected = ' ';
            char predict = ' ';
            for (int i = 0; i < predicted.length; i++) {
                perdictedError += Math.pow(outputs.get(t)[i] - predicted[i], 2);
                if (outputs.get(t)[i] == 1.0) {
                    expected = (char) (i + 'A');
                }
                if (predicted[i] > max) {
                    max = predicted[i];
                    predict = (char) (i + 'A');
                }
            }
            if (predict == expected) {
                success++;
            }
            System.out.println(String.format("Predicted:\t%s\nOutput:\t%s\n", predict, expected));
        }
        System.out.println(String.format("Error at %d is %.12f", maxEpochs, error));
        System.out.println(String.format("Perdicted Error is %.12f", perdictedError * 0.5));
        System.out.println(String.format("success: %d from %d", success, inputs.size() - count));
    }

    public void vectors(MLP object, double learningRate, double percentageUpdate, int epochs) {
        double[][] inputVector = new double[50][4];
        double[][] output = new double[50][1];
        for (int i = 0; i < inputVector.length; i++) {
            for (int j = 0; j < 4; j++) {
                inputVector[i][j] = -1 + Math.random() * 2;
            }
            output[i][0] = Math.sin(inputVector[i][0] - inputVector[i][1] + inputVector[i][2] - inputVector[i][3]);
        }

        int maxEpochs = epochs;
        double error = 0.0;
        MLP mlp = object;
        mlp.randomise();
        for (int e = 0; e < maxEpochs; e++) {
            error = 0.0;
            for (int p = 0; p < inputVector.length - 10; p++) {

                mlp.forward(inputVector[p]);
                error += mlp.backward(output[p]);
                if (Math.random() < percentageUpdate) {
                    mlp.updateWeights(learningRate);
                }

            }
            System.out.println(String.format("Error at %d is %.12f", e, error));
        }

        double perdictedError = 0.0;
        for (int t = inputVector.length - 10; t < inputVector.length; t++) {
            double[] predicted = mlp.forward(inputVector[t]);
            perdictedError += Math.pow(output[t][0] - predicted[0], 2);
            System.out.println(String.format("Predicted:\t%f\nOutput:\t\t%f\n", predicted[0], output[t][0]));
        }
        System.out.println(String.format("Error at %d is %.12f", maxEpochs, error));
        System.out.println(String.format("Perdicted Error is %.12f", perdictedError * 0.5));

    }

    public void xor(MLP object, double leaningRate, double percentageUpdate, int epochs) {
        int maxEpochs = epochs;
        double error = 0.0;

        MLP mlp = object;

        double[][] inputs = new double[][]{
            new double[]{0, 0},
            new double[]{0, 1},
            new double[]{1, 0},
            new double[]{1, 1}
        };
        double[][] outputs = new double[][]{
            new double[]{0},
            new double[]{1},
            new double[]{1},
            new double[]{0}
        };

        mlp.randomise();
        for (int e = 0; e < maxEpochs; e++) {
            error = 0;
            for (int p = 0; p < inputs.length; p++) {
                mlp.forward(inputs[p]);
                error += mlp.backward(outputs[p]);
                if (Math.random() < percentageUpdate) {
                    mlp.updateWeights(leaningRate);
                }
            }
            System.out.println(String.format("Error at %d is %.12f", e, error));
        }

        double perdictedError = 0.0;
        for (int t = 0; t < inputs.length; t++) {
            double[] predicted = mlp.forward(inputs[t]);
            perdictedError += Math.pow(inputs[t][0] - predicted[0], 2);
            System.out.println(String.format("Predicted:\t%f\nOutput:\t\t%f\n", predicted[0], outputs[t][0]));
        }
        System.out.println(String.format("Error at %d is %.12f", maxEpochs, error));
        System.out.println(String.format("Perdicted Error is %.12f", perdictedError * 0.5));
    }
}
