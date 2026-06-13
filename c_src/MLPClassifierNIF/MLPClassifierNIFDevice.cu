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

__global__ void KernelFit(MLPLayer layer,
                          const float* batchX,
                          const float* batchY,
                          int batchCount,
                          float* gradW,
                          float* gradB)
{
    const int tid = threadIdx.x + blockIdx.x * (int)blockDim.x;
    if (tid >= batchCount)
        return;

    // act/delta vivem em memoria local por thread: ficam em L1 e nao varrem a L2,
    // deixando os pesos (reusados por todas as threads) quentes em cache.
    float act[MAX_NEURONS];
    float delta[MAX_NEURONS];

    const int inputSize = layer.m_Layers[0];
    for (int i = 0; i < inputSize; ++i)
        act[i] = batchX[tid * inputSize + i];

    for (int l = 1; l < layer.m_LayerCount; ++l)
    {
        const int prevSize  = layer.m_Layers[l - 1];
        const int currSize  = layer.m_Layers[l];
        const int wOff      = layer.m_WeightOffset[l];
        const int bOff      = layer.m_BiasOffset[l];
        const int prevOff   = layer.m_NeuronOffset[l - 1];
        const int currOff   = layer.m_NeuronOffset[l];
        const bool isOutput = (l == layer.m_LayerCount - 1);

        for (int j = 0; j < currSize; ++j)
        {
            float net = layer.m_Biases[bOff + j];
            for (int i = 0; i < prevSize; ++i)
            {
                net += layer.m_Weights[wOff + i * currSize + j]
                     * act[prevOff + i];
            }
            act[currOff + j] = isOutput ? Sigmoid(net) : ReLU(net);
        }
    }

    const int outL    = layer.m_LayerCount - 1;
    const int outSize = layer.m_Layers[outL];
    const int outOff  = layer.m_NeuronOffset[outL];
    const float y     = batchY[tid];

    for (int j = 0; j < outSize; ++j)
        delta[outOff + j] = act[outOff + j] - y;

    for (int l = outL - 1; l >= 1; --l)
    {
        const int currSize = layer.m_Layers[l];
        const int nextSize = layer.m_Layers[l + 1];
        const int currOff  = layer.m_NeuronOffset[l];
        const int nextOff  = layer.m_NeuronOffset[l + 1];
        const int nextWOff = layer.m_WeightOffset[l + 1];

        for (int j = 0; j < currSize; ++j)
        {
            float sum = 0.f;
            for (int k = 0; k < nextSize; ++k)
                sum += layer.m_Weights[nextWOff + j * nextSize + k]
                     * delta[nextOff + k];

            delta[currOff + j] = sum * ReLUDeriv(act[currOff + j]);
        }
    }

    for (int l = 1; l < layer.m_LayerCount; ++l)
    {
        const int prevSize = layer.m_Layers[l - 1];
        const int currSize = layer.m_Layers[l];
        const int wOff     = layer.m_WeightOffset[l];
        const int bOff     = layer.m_BiasOffset[l];
        const int prevOff  = layer.m_NeuronOffset[l - 1];
        const int currOff  = layer.m_NeuronOffset[l];

        for (int j = 0; j < currSize; ++j)
        {
            const float d = delta[currOff + j];
            atomicAdd(&gradB[bOff + j], d);

            for (int i = 0; i < prevSize; ++i)
            {
                const float aPrev = act[prevOff + i];
                atomicAdd(&gradW[wOff + i * currSize + j], d * aPrev);
            }
        }
    }
}

__global__ void KernelPredict(MLPLayer layer,
                              const float* batchX,
                              float* results,
                              int batchCount)
{
    const int tid = threadIdx.x + blockIdx.x * (int)blockDim.x;
    if (tid >= batchCount)
        return;

    float act[MAX_NEURONS];

    const int inputSize = layer.m_Layers[0];
    for (int i = 0; i < inputSize; ++i)
        act[i] = batchX[tid * inputSize + i];

    for (int l = 1; l < layer.m_LayerCount; ++l)
    {
        const int prevSize  = layer.m_Layers[l - 1];
        const int currSize  = layer.m_Layers[l];
        const int wOff      = layer.m_WeightOffset[l];
        const int bOff      = layer.m_BiasOffset[l];
        const int prevOff   = layer.m_NeuronOffset[l - 1];
        const int currOff   = layer.m_NeuronOffset[l];
        const bool isOutput = (l == layer.m_LayerCount - 1);

        for (int j = 0; j < currSize; ++j)
        {
            float net = layer.m_Biases[bOff + j];
            for (int i = 0; i < prevSize; ++i)
                net += layer.m_Weights[wOff + i * currSize + j]
                     * act[prevOff + i];

            act[currOff + j] = isOutput ? Sigmoid(net) : ReLU(net);
        }
    }

    const int outL    = layer.m_LayerCount - 1;
    const int outSize = layer.m_Layers[outL];
    const int outOff  = layer.m_NeuronOffset[outL];
    for (int j = 0; j < outSize; ++j)
        results[tid * outSize + j] = act[outOff + j];
}

void LaunchFitKernel(const MLPLayer& layer,
                     const float* batchX,
                     const float* batchY,
                     int batchCount,
                     float* gradW,
                     float* gradB)
{
    const int grid = (batchCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    KernelFit<<<grid, CUDA_BLOCK_SIZE>>>(
        layer, batchX, batchY, batchCount, gradW, gradB);
}

void LaunchPredictKernel(const MLPLayer& layer,
                         const float* batchX,
                         float* results,
                         int batchCount)
{
    const int grid = (batchCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    KernelPredict<<<grid, CUDA_BLOCK_SIZE>>>(layer, batchX, results, batchCount);
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
