//
// Created by Rahul Chunduru on 12/22.
//

#include "pipeline/pipeline_gpu.h"
#include "gpu_kernel.h"

__global__
void  gpuCompute(Pipeline pipeline_, int gpu_id_, shared_ptr<Batch> batch) {
    pipeline_->model_->device_models_[gpu_id_].get()->train_batch(batch, ((PipelineGPU *)pipeline_)->pipeline_options_->gpu_model_average);
}

void launchKernel(void* pipeline_, int gpu_id_, void* batch) {
    gpuCompute<<<1, 64>>>((Pipeline*)pipeline_, gpu_id_, (shared_ptr<Batch>)batch);
}
