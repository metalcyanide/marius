//
// Created by Rahul Chunduru on 12/22.
//

void launchKernel(void* pipeline_, int gpu_id_, void* batch);

void transferDataToDevice(void* pipeline_, void* batch, int queue_choice)
