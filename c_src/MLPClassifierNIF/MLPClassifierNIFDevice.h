#pragma once

#include "MLPClassifierNIF.h"

static const int CUDA_BLOCK_SIZE = 256;
static const int MAX_NEURONS = 512;

void LaunchFitKernel(const MLPLayer& layer, const float* batchX, const float* batchY, int batchCount, float* gradW, float* gradB);
void LaunchPredictKernel(const MLPLayer& layer, const float* batchX, float* results, int batchCount);
void LaunchUpdateWeightsKernel(float* weights, float* biases, const float* gradW, const float* gradB, int weightCount, int biasCount, float learnRate, int batchCount);

