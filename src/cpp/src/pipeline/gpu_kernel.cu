//
// Created by Rahul Chunduru on 12/22.
//

#include "pipeline/pipeline_gpu.h"
#include "pipeline/gpu_kernel.h"

__global__
void  gpuCompute(void* pipeline_, int gpu_id_, void* batch) {
    // ((PipelineGPU*)pipeline_)->model_->device_models_[gpu_id_].get()->train_batch(*((shared_ptr<Batch>*)batch), ((PipelineGPU *)pipeline_)->pipeline_options_->gpu_model_average);
}

__global__
void  transferBatchToDevice(void* pipeline_, void* batch, int queue_choice) {
    *((shared_ptr<Batch>*)batch)->to((PipelineGPU*)pipeline_->model_->device_models_[queue_choice]->device_)

}

void launchKernel(void* pipeline_, int gpu_id_, void* batch) {
    gpuCompute<<<1, 64>>>(pipeline_, batch, queue_choice);
}

void transferDataToDevice(void* pipeline_, void* batch, int queue_choice) {
    transferBatchToDevice<<<1, 64>>>(pipeline_, batch, queue_choice);
}
