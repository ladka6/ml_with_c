#include <stdio.h>
#include "Headers/nn.h"

#define SAD_PREDICTION_IDX 0
#define SICK_PREDICTION_IDX 1
#define ACTIVE_PREDICTION_IDX 2

#define IN_LEN 3
#define OUT_LEN 3

/*
double predicted_results[3];
double input_to_hidden_weight[HID_LEN][IN_LEN] = {
        {-2,9.5,2.01},
        {-0.8,7.2,6.3},
        {-0.5,0.45,0.9},
        };
double hidden_to_output_weight[OUT_LEN][HID_LEN] = {
        {-1,1.15,0.11},
        {-0.1,4.2,2.6},
        {-0.76,4.5,0.29},
};
double inputs[IN_LEN] = {30,87,119};
double expected_values[OUT_LEN] = {600,10,-90};
 */
double weight = 0.5;
double input = 0.5;
double expected_amount = 0.8;
double step_amount = 0.001;
int main(void) {
    /*
    hidden_layer_nn(inputs, IN_LEN, HID_LEN,input_to_hidden_weight,OUT_LEN,hidden_to_output_weight,predicted_results);
    printf("Sad prediction : %f \n",predicted_results[SAD_PREDICTION_IDX]);
    printf("Error for sad : %f \n", find_err_simple(predicted_results[SAD_PREDICTION_IDX],expected_values[SAD_PREDICTION_IDX]));

    printf("Sick prediction : %f \n",predicted_results[SICK_PREDICTION_IDX]);
    printf("Error for sick : %f \n", find_err_simple(predicted_results[SICK_PREDICTION_IDX],expected_values[SICK_PREDICTION_IDX]));

    printf("Active prediction : %f \n",predicted_results[ACTIVE_PREDICTION_IDX]);
    printf("Error for sick : %f \n", find_err_simple(predicted_results[ACTIVE_PREDICTION_IDX],expected_values[ACTIVE_PREDICTION_IDX]));
     */
    brute_force_learning(input, weight, expected_amount, step_amount, 1300);
    return 0;
}
