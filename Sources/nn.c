//
// Created by Ege Erdal on 23.05.2024.
//
#include "../Headers/nn.h"
#include <stdio.h>

double single_in_single_out(double input, double weight) {
    return (input * weight);
}

double weighted_sum(double * input, double * weight, int LEN) {
    double output = 0.0;

    for(int i = 0; i<LEN; i++) {
        output += input[i]* weight[i];
    }
    return output;
}

double multiple_input_single_output(double * input, double * weight, int LEN) {
    double predicted_value;

    predicted_value = weighted_sum(input, weight, LEN);

    return predicted_value;
}

void elementwise_multiple(double input_scalar, const double * weight_vector, double * output_vector, int LEN){
    for(int i = 0; i < LEN; i++) {
        output_vector[i] = input_scalar * weight_vector[i];
    }
}

void single_in_multiple_out_nn(double input_scalar, double * weight_vector, double * output_vector, int LEN){
    elementwise_multiple(input_scalar,weight_vector, output_vector, LEN);
}

void matrix_vector_multiply(double * input_vector,
                            int INPUT_LEN,
                            double * output_vector,
                            int OUTPUT_LEN,
                            double weight_matrix[OUTPUT_LEN][INPUT_LEN]) {
    for(int k = 0; k < OUTPUT_LEN; k++) {
        for(int i = 0; i<INPUT_LEN;i++) {
            output_vector[k] += input_vector[i]*weight_matrix[k][i];
        }
    }
}

void multiple_input_multiple_output_nn(double * input_vector,
                                       int INPUT_LEN,
                                       double * output_vector,
                                       int OUTPUT_LEN,
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN]) {
    matrix_vector_multiply(input_vector,INPUT_LEN,output_vector,OUTPUT_LEN,weight_matrix);
}


