#include<stdio.h>
#include<cuda.h>

__global__ void AdamW(float *weights, float *gradients, float *momentum, float *velocity, float lambda,float lr,
                        int M, int N, float beta_1, float beta_2, float eps, int timestep ){

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if(row < M && col <N){

        float curr_momentum = momentum[row * N + col];
        float curr_velocity = velocity[row *N + col];
        float curr_gradient = gradients[row *N + col];
        float curr_weight = weights[row * N + col];

        float updated_momentum = beta_1 * curr_momentum + (1.0f - beta_1) * curr_gradient;
        float updated_velocity = beta_2 * curr_velocity + (1.0f - beta_2) * curr_gradient * curr_gradient;

        float m_t_corrected = updated_momentum / (1.0f - powf(beta_1, timestep));
        float v_t_corrected = updated_velocity / (1.0f - powf(beta_2, timestep));

        float weight_decay_term = lr * lambda * curr_weight;
        float gradient_term = lr * m_t_corrected / (sqrtf(v_t_corrected + eps));

        weights[row * N + col] = curr_weight  - gradient_term - weight_decay_term;
        momentum[row *N + col ] = updated_momentum;
        velocity[row *N + col] = updated_velocity;
    }

}
