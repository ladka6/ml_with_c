#include <stdio.h>
#include "Headers/nn.h"

#define Sad 0.9

#define TEMPERATURE_PREDICTION_IDX 0
#define HUMIDITY_PREDICTION_IDX 1
#define AIR_QUALITY_PREDICTION_IDX 2

#define IN_LEN 3
#define OUT_LEN 3

double predicted_results[3];
double weights[OUT_LEN][IN_LEN] = {
        {-2,9.5,2.01},
        {-0.8,7.2,6.3},
        {-0.5,0.45,0.9},
        };
double inputs[IN_LEN] = {30,87,119};
int main(void) {
    multiple_input_multiple_output_nn(inputs,IN_LEN,predicted_results,OUT_LEN, weights);
    printf("Sad prediction : %f \n",predicted_results[0]);
}
