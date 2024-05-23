//
// Created by Ege Erdal on 23.05.2024.
//

#include <math.h>

#ifndef C_TEST_NN_H
#define C_TEST_NN_H
#define HID_LEN 3

double single_in_single_out(double input, double weight);
void single_in_multiple_out_nn(double input_scalar, double * weight_vector, double * output_vector, int LEN);
void multiple_input_multiple_output_nn(double * input_vector,
                                       int INPUT_LEN,
                                       double * output_vector,
                                       int OUTPUT_LEN,
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN]);
double hidden_pred_vector[HID_LEN];
void hidden_layer_nn(double * input_vector,
                     int INPUT_LEN,
                     int HIDDEN_LEN,
                     double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
                     int OUTPUT_LEN,
                     double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN],
                     double * output_vector);
double find_err(double input, double weight, double expected_value);
double find_err_simple(double yhat, double y);
void brute_force_learning(double input,
                          double weight,
                          double expected_value,
                          double step_amount,
                          int epochs);

#endif //C_TEST_NN_H
