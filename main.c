#include <stdio.h>
#include "Headers/nn.h"

#define NUM_OF_FEATURES 2
#define NUM_OF_EXAMPLES 3
#define NUM_OF_HID_NODES 3
#define NUM_OF_OUT_NODES 1

double raw_x[NUM_OF_FEATURES][NUM_OF_EXAMPLES] = {
    {2, 5, 1},
    {8, 5, 8},
};

double raw_y[1][NUM_OF_EXAMPLES] = {
    {200, 90, 190},
};

double train_x[NUM_OF_FEATURES][NUM_OF_EXAMPLES];
double train_y[1][NUM_OF_EXAMPLES];

// Input layer to hidden layer weight matrix
double syn0[NUM_OF_HID_NODES][NUM_OF_FEATURES];

// Hidden layer to output layer weight matrix
double syn1[NUM_OF_HID_NODES];

double train_x_eg1[NUM_OF_FEATURES];
double train_y_eg1;
double z1_eg_1[NUM_OF_HID_NODES];
double a1_eg_1[NUM_OF_HID_NODES];
double z2_eg1;
double yhat_eg1;

int main(void)
{
    // Normalize x and y
    normalize_data_2D(NUM_OF_FEATURES, NUM_OF_EXAMPLES, raw_x, train_x);
    normalize_data_2D(1, NUM_OF_EXAMPLES, raw_y, train_y);

    train_x_eg1[0] = train_x[0][0];
    train_x_eg1[1] = train_x[1][0];
    train_y_eg1 = train_y[0][0];

    printf("Train x eg 1 : %f %f \n", train_x_eg1[0], train_x_eg1[1]);
    printf("Train y eg 1 : %f \n", train_y_eg1);

    // Initialize syn0 and syn1 weights
    weight_random_initialization(NUM_OF_HID_NODES, NUM_OF_FEATURES, syn0);
    for (int i = 0; i < NUM_OF_HID_NODES; i++)
    {
        for (int j = 0; j < NUM_OF_FEATURES; j++)
        {
            printf(" %f ", syn0[i][j]);
        }
        printf("\n");
    }

    weight_random_initialization_1d(syn1, NUM_OF_HID_NODES);
    for (int i = 0; i < NUM_OF_HID_NODES; i++)
    {
        printf("Syn1 : %f ", syn1[i]);
    }

    multiple_input_multiple_output_nn(train_x_eg1, NUM_OF_FEATURES, z1_eg_1, NUM_OF_HID_NODES, syn0);
    printf("\n");

    printf("Z1 : [%f %f %f] \n", z1_eg_1[0], z1_eg_1[1], z1_eg_1[2]);
    printf("\n");

    // Compute a1
    vetor_sigmoid(z1_eg_1, a1_eg_1, NUM_OF_HID_NODES);
    printf("A1 : [%f %f %f] \n", a1_eg_1[0], a1_eg_1[1], a1_eg_1[2]);
    printf("\n");

    // Compute z2
    z2_eg1 = multiple_input_single_output(a1_eg_1, syn1, NUM_OF_HID_NODES);
    printf("Z2 : %f \n", z2_eg1);
    printf("\n");

    // Compute yhat
    yhat_eg1 = sigmoid(z2_eg1);
    printf("Yhat : %f \n", yhat_eg1);
    printf("\n");

    // weight_random_initialization(NUM_OF_HID_NODES, NUM_OF_FEATURES, syn0);
    // weight_random_initialization(NUM_OF_OUT_NODES, NUM_OF_HID_NODES, syn1);

    // printf("Syn 0 \n");
    // for (int i = 0; i < NUM_OF_HID_NODES; ++i)
    // {
    //     for (int j = 0; j < NUM_OF_FEATURES; ++j)
    //     {
    //         printf(" %f ", syn0[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("Syn 1 \n");
    // for (int i = 0; i < NUM_OF_OUT_NODES; ++i)
    // {
    //     for (int j = 0; j < NUM_OF_HID_NODES; ++j)
    //     {
    //         // printf(" %f ", syn1[i][j]);
    //     }
    //     printf("\n");
    // }
}
