#include "MLPClassifierNIFDevice.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ReLU nas camadas ocultas: gradiente constante (1) para entradas positivas.
__device__ __forceinline__ float ReLU(float x) { return x > 0.f ? x : 0.f; }
__device__ __forceinline__ float ReLUDeriv(float a) { return a > 0.f ? 1.f : 0.f; }

// Sigmoide na saida: mapeia R para (0,1), interpretavel como P(y=1|x).
__device__ __forceinline__ float Sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void KernelUpdateWeights(float* weights, float* biases,
                                    const float* gradW, const float* gradB,
                                    int weightCount, int biasCount,
                                    float learnRate, int batchCount)
{
    const int tid   = threadIdx.x + blockIdx.x * (int)blockDim.x;
    const float scl = learnRate / (float)batchCount;

    if (tid < weightCount)
        weights[tid] -= scl * gradW[tid];

    if (tid < biasCount)
        biases[tid] -= scl * gradB[tid];
}

__device__ void FitDevice(MLPLayer* layer,
                          const float* batchX,
                          const float* batchY,
                          int batchCount,
                          float* act,
                          float* delta,
                          float* gradW,
                          float* gradB)
{
    const int tid = threadIdx.x + blockIdx.x * (int)blockDim.x;
    if (tid >= batchCount)
        return;

    const int inputSize = layer->m_Layers[0];
    for (int i = 0; i < inputSize; ++i)
        act[(layer->m_NeuronOffset[0] + i) * batchCount + tid] = batchX[i * batchCount + tid];

    for (int l = 1; l < layer->m_LayerCount; ++l)
    {
        const int prevSize  = layer->m_Layers[l - 1];
        const int currSize  = layer->m_Layers[l];
        const bool isOutput = (l == layer->m_LayerCount - 1);

        for (int j = 0; j < currSize; ++j)
        {
            float net = layer->m_Biases[l][j];
            for (int i = 0; i < prevSize; ++i)
            {
                net += layer->m_Weights[l][i * currSize + j]
                     * act[(layer->m_NeuronOffset[l - 1] + i) * batchCount + tid];
            }
            act[(layer->m_NeuronOffset[l] + j) * batchCount + tid] =
                isOutput ? Sigmoid(net) : ReLU(net);
        }
    }

    const int outL    = layer->m_LayerCount - 1;
    const int outSize = layer->m_Layers[outL];
    const float y     = batchY[tid];

    for (int j = 0; j < outSize; ++j)
    {
        const float yhat = act[(layer->m_NeuronOffset[outL] + j) * batchCount + tid];
        delta[(layer->m_NeuronOffset[outL] + j) * batchCount + tid] = yhat - y;
    }

    for (int l = outL - 1; l >= 1; --l)
    {
        const int currSize = layer->m_Layers[l];
        const int nextSize = layer->m_Layers[l + 1];

        for (int j = 0; j < currSize; ++j)
        {
            float sum = 0.f;
            for (int k = 0; k < nextSize; ++k)
                sum += layer->m_Weights[l + 1][j * nextSize + k]
                     * delta[(layer->m_NeuronOffset[l + 1] + k) * batchCount + tid];

            const float a = act[(layer->m_NeuronOffset[l] + j) * batchCount + tid];
            delta[(layer->m_NeuronOffset[l] + j) * batchCount + tid] = sum * ReLUDeriv(a);
        }
    }

    for (int l = 1; l < layer->m_LayerCount; ++l)
    {
        const int prevSize = layer->m_Layers[l - 1];
        const int currSize = layer->m_Layers[l];
        const int wOff     = layer->m_WeightOffset[l];
        const int bOff     = layer->m_BiasOffset[l];

        for (int j = 0; j < currSize; ++j)
        {
            const float d = delta[(layer->m_NeuronOffset[l] + j) * batchCount + tid];
            atomicAdd(&gradB[bOff + j], d);

            for (int i = 0; i < prevSize; ++i)
            {
                const float aPrev = act[(layer->m_NeuronOffset[l - 1] + i) * batchCount + tid];
                atomicAdd(&gradW[wOff + i * currSize + j], d * aPrev);
            }
        }
    }
}

__device__ void PredictDevice(MLPLayer* layer,
                              const float* batchX,
                              float* results,
                              int batchCount,
                              float* act)
{
    const int tid = threadIdx.x + blockIdx.x * (int)blockDim.x;
    if (tid >= batchCount)
        return;

    const int inputSize = layer->m_Layers[0];
    for (int i = 0; i < inputSize; ++i)
        act[(layer->m_NeuronOffset[0] + i) * batchCount + tid] = batchX[i * batchCount + tid];

    for (int l = 1; l < layer->m_LayerCount; ++l)
    {
        const int prevSize  = layer->m_Layers[l - 1];
        const int currSize  = layer->m_Layers[l];
        const bool isOutput = (l == layer->m_LayerCount - 1);

        for (int j = 0; j < currSize; ++j)
        {
            float net = layer->m_Biases[l][j];
            for (int i = 0; i < prevSize; ++i)
                net += layer->m_Weights[l][i * currSize + j]
                     * act[(layer->m_NeuronOffset[l - 1] + i) * batchCount + tid];

            act[(layer->m_NeuronOffset[l] + j) * batchCount + tid] =
                isOutput ? Sigmoid(net) : ReLU(net);
        }
    }

    const int outL    = layer->m_LayerCount - 1;
    const int outSize = layer->m_Layers[outL];
    for (int j = 0; j < outSize; ++j)
        results[j * batchCount + tid] =
            act[(layer->m_NeuronOffset[outL] + j) * batchCount + tid];
}

__global__ void KernelFit(MLPLayer* layer,
                          const float* batchX,
                          const float* batchY,
                          int batchCount,
                          float* act,
                          float* delta,
                          float* gradW,
                          float* gradB)
{
    FitDevice(layer, batchX, batchY, batchCount, act, delta, gradW, gradB);
}

__global__ void KernelPredict(MLPLayer* layer,
                              const float* batchX,
                              float* results,
                              int batchCount,
                              float* act)
{
    PredictDevice(layer, batchX, results, batchCount, act);
}

void LaunchFitKernel(MLPLayer* layer,
                     const float* batchX,
                     const float* batchY,
                     int batchCount,
                     float* act,
                     float* delta,
                     float* gradW,
                     float* gradB)
{
    const int grid = (batchCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    KernelFit<<<grid, CUDA_BLOCK_SIZE>>>(layer, batchX, batchY, batchCount, act, delta, gradW, gradB);
}

void LaunchPredictKernel(MLPLayer* layer,
                         const float* batchX,
                         float* results,
                         int batchCount,
                         float* act)
{
    const int grid = (batchCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    KernelPredict<<<grid, CUDA_BLOCK_SIZE>>>(layer, batchX, results, batchCount, act);
}

void LaunchUpdateWeightsKernel(float* weights,
                               float* biases,
                               const float* gradW,
                               const float* gradB,
                               int weightCount,
                               int biasCount,
                               float learnRate,
                               int batchCount)
{
    const int maxCount = weightCount > biasCount ? weightCount : biasCount;
    const int grid = (maxCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    KernelUpdateWeights<<<grid, CUDA_BLOCK_SIZE>>>(
        weights, biases, gradW, gradB, weightCount, biasCount, learnRate, batchCount);
}
