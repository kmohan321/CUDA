// #include "solve.h"
#include<stdio.h>
#include <cuda_runtime.h>

__global__ void MA_flock_simulation(const float *agents, float *agents_next, int N){

    int agent_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(agent_id >=N) return;
    int agent_pos = agent_id *4;
    float agent_x = agents[agent_pos + 0];
    float agent_y = agents[agent_pos + 1];
    float agent_vx = agents[agent_pos + 2];
    float agent_vy = agents[agent_pos + 3];

    float radius_sq = 25.0f;

    int neighbour_count = 0;
    float v_nextx = 0.0f;
    float v_nexty = 0.0f;
    for(int j =0; j<N; j++){

        if(agent_id!=j){ //skiping the self element

            float agent_nextx = agents[4*j + 0];
            float agent_nexty = agents[4*j + 1];

            float x_distance = agent_x - agent_nextx;
            float y_distance = agent_y - agent_nexty;
            float x_sq = x_distance * x_distance;
            float y_sq = y_distance * y_distance;

            if((x_sq + y_sq) <= radius_sq ){
                v_nextx += agents[4*j + 2];
                v_nexty += agents[4*j + 3];
                neighbour_count += 1;
            }
        } 
    }
    float updated_vx = agent_vx;
    float updated_vy = agent_vy;
    if(neighbour_count>0){
        float vavg_next_x = v_nextx/neighbour_count;
        float vavg_next_y = v_nexty/neighbour_count;

        float updated_vx = agent_vx + 0.05f * (vavg_next_x - agent_vx);
        float updated_vy = agent_vy + 0.05f * (vavg_next_y - agent_vy);
    }

    float updated_x = updated_vx + agent_x;
    float updated_y = updated_vy + agent_y;

    agents_next[agent_pos + 0] = updated_x;
    agents_next[agent_pos + 1] = updated_y;
    agents_next[agent_pos + 2] = updated_vx;
    agents_next[agent_pos + 3] = updated_vy;
}

// agents, agents_next are device pointers
void solve(const float* agents, float* agents_next, int N) {

    int blocksize = 256;
    int grid = N + 256-1 / 256;

    MA_flock_simulation<<<grid,blocksize>>>(agents, agents_next, N);
    
}
