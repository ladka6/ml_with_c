//
// Created by Ege Erdal on 23.05.2024.
//

#ifndef C_TEST_NN_H
#define C_TEST_NN_H
double single_in_single_out(double input, double weight);
void single_in_multiple_out_nn(double input_scalar, double * weight_vector, double * output_vector, int LEN);
void multiple_input_multiple_output_nn(double * input_vector,
                                       int INPUT_LEN,
                                       double * output_vector,
                                       int OUTPUT_LEN,
                                       double weight_matrix[OUTPUT_LEN][INPUT_LEN]);
#endif //C_TEST_NN_H
