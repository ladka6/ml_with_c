//
// Created by Ege Erdal on 23.05.2024.
//
#include "../Headers/nn.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double single_in_single_out(double input, double weight)
{
    return (input * weight);
}

double weighted_sum(double *input, double *weight, int LEN)
{
    double output = 0.0;

    for (int i = 0; i < LEN; i++)
    {
        output += input[i] * weight[i];
    }
    return output;
}

double multiple_input_single_output(double *input, double *weight, int LEN)
{
    double predicted_value;

    predicted_value = weighted_sum(input, weight, LEN);

    return predicted_value;
}

void elementwise_multiple(double input_scalar, const double *weight_vector, double *output_vector, int LEN)
{
    for (int i = 0; i < LEN; i++)
    {
        output_vector[i] = input_scalar * weight_vector[i];
    }
}

void single_in_multiple_out_nn(double input_scalar, double *weight_vector, double *output_vector, int LEN)
{
    elementwise_multiple(input_scalar, weight_vector, output_vector, LEN);
}

void matrix_vector_multiply(double *input_vector,
                            int INPUT_LEN,
                            double *output_vector,
                            int OUTPUT_LEN,
                            double weight_matrix[OUTPUT_LEN][INPUT_LEN])
{
    for (int k = 0; k < OUTPUT_LEN; k++)
    {
        for (int i = 0; i < INPUT_LEN; i++)
        {
            output_vector[k] += input_vector[i] * weight_matrix[k][i];
        }
    }
}

void multiple_input_multiple_output_nn(double *input_vector,
                                       int INPUT_LEN,
                                       double *output_vector,
                                       int OUTPUT_LEN,
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN])
{
    matrix_vector_multiply(input_vector, INPUT_LEN, output_vector, OUTPUT_LEN, weight_matrix);
}

void hidden_layer_nn(double *input_vector,
                     int INPUT_LEN,
                     int HIDDEN_LEN,
                     double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
                     int OUTPUT_LEN,
                     double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN],
                     double *output_vector)
{

    matrix_vector_multiply(input_vector, INPUT_LEN, hidden_pred_vector, HIDDEN_LEN, in_to_hid_weights);
    matrix_vector_multiply(hidden_pred_vector, HIDDEN_LEN, output_vector, OUTPUT_LEN, hid_to_out_weights);
}

double find_err(double input, double weight, double expected_value)
{
    return powf((input * weight) - expected_value, 2);
}

double find_err_simple(double yhat, double y)
{
    return powf((yhat - y), 2);
}

void brute_force_learning(double input,
                          double weight,
                          double expected_value,
                          double step_amount,
                          int epochs)
{
    double prediction, error;
    double up_prediction, up_error, down_prediction, down_error;

    for (int i = 0; i < epochs; i++)
    {
        prediction = input * weight;
        error = powf((prediction - expected_value), 2);
        printf("Error : %f Prediction : %f \n", error, prediction);

        up_prediction = input * (weight + step_amount);
        up_error = powf((expected_value - up_prediction), 2);

        down_error = input * (weight - step_amount);
        down_error = powf((expected_value - down_prediction), 2);

        if (down_error < up_error)
            weight = weight - step_amount;
        if (down_error > up_error)
            weight = weight + step_amount;
    }
}

void normalize_data(double *input_vector, double *output_vector, int LEN)
{
    int i;
    /*Find max*/
    double max = input_vector[0];

    for (i = 1; i < LEN; i++)
    {
        if (input_vector[i] > max)
        {
            max = input_vector[i];
        }
    }

    // Normalize
    for (i = 0; i < LEN; ++i)
    {
        output_vector[i] = input_vector[i] / max;
    }
}

void weight_random_initialization(int HIDDEN_LEN,
                                  int INPUT_LEN,
                                  double weights_matrix[HIDDEN_LEN][INPUT_LEN])
{
    double d_rand;
    // Seed random number generator
    srand(2);
    for (int i = 0; i < HIDDEN_LEN; ++i)
    {
        for (int j = 0; j < INPUT_LEN; ++j)
        {
            // Generate random numbers between 0 and 1;
            d_rand = (rand() % 10);
            d_rand /= 10;

            weights_matrix[i][j] = d_rand;
        }
    }
}

void normalize_data_2D(int ROW, int COL, double input_matrix[ROW][COL], double output_matrix[ROW][COL])
{
    int i, j;
    /*Find max*/
    double max = input_matrix[0][0];

    for (i = 0; i < ROW; i++)
    {
        for (j = 0; j < COL; j++)
        {
            if (input_matrix[i][j] > max)
            {
                max = input_matrix[i][j];
            }
        }
    }

    // Normalize
    for (i = 0; i < ROW; ++i)
    {
        for (j = 0; j < COL; j++)
        {
            output_matrix[i][j] = input_matrix[i][j] / max;
        }
    }
}

void weight_random_initialization_1d(double *output_vector, int LEN)
{
    double d_rand;
    // Seed random number generator
    srand(2);
    for (int i = 0; i < LEN; ++i)
    {
        // Generate random numbers between 0 and 1;
        d_rand = (rand() % 10);
        d_rand /= 10;

        output_vector[i] = d_rand;
    }
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

void vetor_sigmoid(double *input_vector, double *output_vector, int LEN)
{
    for (int i = 0; i < LEN; i++)
    {
        output_vector[i] = sigmoid(input_vector[i]);
    }
}