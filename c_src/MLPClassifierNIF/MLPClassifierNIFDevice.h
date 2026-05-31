#pragma once

#include "MLPClassifierNIF.h"

static const int CUDA_BLOCK_SIZE = 256;

void LaunchFitKernel(MLPLayer* layer,
                     const float* batchX,
                     const float* batchY,
                     int batchCount,
                     float* act,
                     float* delta,
                     float* gradW,
                     float* gradB);

void LaunchPredictKernel(MLPLayer* layer,
                         const float* batchX,
                         float* results,
                         int batchCount,
                         float* act);

void LaunchUpdateWeightsKernel(float* weights,
                               float* biases,
                               const float* gradW,
                               const float* gradB,
                               int weightCount,
                               int biasCount,
                               float learnRate,
                               int batchCount);
