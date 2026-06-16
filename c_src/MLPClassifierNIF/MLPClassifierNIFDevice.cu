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

// Uma amostra por BLOCO; as threads do bloco cooperam sobre os neuronios de cada
// camada. act/delta vivem em shared memory (uma copia por bloco, indexada por
// m_NeuronOffset), nao em arrays por thread. Com indexacao dinamica, aqueles arrays
// estouravam os registradores e iam para memoria local (DRAM), tornando o kernel
// limitado por latencia de memoria. Em shared memory ficam on-chip, e distribuir as
// amostras em blocos (em vez de 1 thread cada) melhora a ocupacao da GPU.
__global__ void KernelFit(MLPLayer layer,
                          const float* batchX,
                          const float* batchY,
                          int batchCount,
                          float* gradW,
                          float* gradB)
{
    const int sample = blockIdx.x;
    if (sample >= batchCount)
        return;

    const int tid      = threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const int totalN   = layer.m_TotalNeurons;

    extern __shared__ float shared[];
    float* act   = shared;          // [totalN]
    float* delta = shared + totalN; // [totalN]

    const int inputSize = layer.m_Layers[0];
    for (int i = tid; i < inputSize; i += nthreads)
        act[i] = batchX[sample * inputSize + i];
    __syncthreads();

    for (int l = 1; l < layer.m_LayerCount; ++l)
    {
        const int prevSize  = layer.m_Layers[l - 1];
        const int currSize  = layer.m_Layers[l];
        const int wOff      = layer.m_WeightOffset[l];
        const int bOff      = layer.m_BiasOffset[l];
        const int prevOff   = layer.m_NeuronOffset[l - 1];
        const int currOff   = layer.m_NeuronOffset[l];
        const bool isOutput = (l == layer.m_LayerCount - 1);

        for (int j = tid; j < currSize; j += nthreads)
        {
            float net = layer.m_Biases[bOff + j];
            for (int i = 0; i < prevSize; ++i)
                net += layer.m_Weights[wOff + i * currSize + j] * act[prevOff + i];

            act[currOff + j] = isOutput ? Sigmoid(net) : ReLU(net);
        }
        __syncthreads();
    }

    const int outL    = layer.m_LayerCount - 1;
    const int outSize = layer.m_Layers[outL];
    const int outOff  = layer.m_NeuronOffset[outL];
    const float y     = batchY[sample];

    for (int j = tid; j < outSize; j += nthreads)
        delta[outOff + j] = act[outOff + j] - y;
    __syncthreads();

    for (int l = outL - 1; l >= 1; --l)
    {
        const int currSize = layer.m_Layers[l];
        const int nextSize = layer.m_Layers[l + 1];
        const int currOff  = layer.m_NeuronOffset[l];
        const int nextOff  = layer.m_NeuronOffset[l + 1];
        const int nextWOff = layer.m_WeightOffset[l + 1];

        for (int j = tid; j < currSize; j += nthreads)
        {
            float sum = 0.f;
            for (int k = 0; k < nextSize; ++k)
                sum += layer.m_Weights[nextWOff + j * nextSize + k] * delta[nextOff + k];

            delta[currOff + j] = sum * ReLUDeriv(act[currOff + j]);
        }
        __syncthreads();
    }

    for (int l = 1; l < layer.m_LayerCount; ++l)
    {
        const int prevSize = layer.m_Layers[l - 1];
        const int currSize = layer.m_Layers[l];
        const int wOff     = layer.m_WeightOffset[l];
        const int bOff     = layer.m_BiasOffset[l];
        const int prevOff  = layer.m_NeuronOffset[l - 1];
        const int currOff  = layer.m_NeuronOffset[l];

        for (int j = tid; j < currSize; j += nthreads)
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
    const int sample = blockIdx.x;
    if (sample >= batchCount)
        return;

    const int tid      = threadIdx.x;
    const int nthreads = (int)blockDim.x;

    extern __shared__ float act[]; // [m_TotalNeurons]

    const int inputSize = layer.m_Layers[0];
    for (int i = tid; i < inputSize; i += nthreads)
        act[i] = batchX[sample * inputSize + i];
    __syncthreads();

    for (int l = 1; l < layer.m_LayerCount; ++l)
    {
        const int prevSize  = layer.m_Layers[l - 1];
        const int currSize  = layer.m_Layers[l];
        const int wOff      = layer.m_WeightOffset[l];
        const int bOff      = layer.m_BiasOffset[l];
        const int prevOff   = layer.m_NeuronOffset[l - 1];
        const int currOff   = layer.m_NeuronOffset[l];
        const bool isOutput = (l == layer.m_LayerCount - 1);

        for (int j = tid; j < currSize; j += nthreads)
        {
            float net = layer.m_Biases[bOff + j];
            for (int i = 0; i < prevSize; ++i)
                net += layer.m_Weights[wOff + i * currSize + j] * act[prevOff + i];

            act[currOff + j] = isOutput ? Sigmoid(net) : ReLU(net);
        }
        __syncthreads();
    }

    const int outL    = layer.m_LayerCount - 1;
    const int outSize = layer.m_Layers[outL];
    const int outOff  = layer.m_NeuronOffset[outL];
    for (int j = tid; j < outSize; j += nthreads)
        results[sample * outSize + j] = act[outOff + j];
}

void LaunchFitKernel(const MLPLayer& layer,
                     const float* batchX,
                     const float* batchY,
                     int batchCount,
                     float* gradW,
                     float* gradB)
{
    // Um bloco por amostra; shared memory para act + delta (2 * total de neuronios).
    const int grid = batchCount;
    const size_t sharedBytes = sizeof(float) * 2 * (size_t)layer.m_TotalNeurons;
    KernelFit<<<grid, CUDA_BLOCK_SIZE, sharedBytes>>>(
        layer, batchX, batchY, batchCount, gradW, gradB);
}

void LaunchPredictKernel(const MLPLayer& layer,
                         const float* batchX,
                         float* results,
                         int batchCount)
{
    // Um bloco por amostra; shared memory para act (total de neuronios).
    const int grid = batchCount;
    const size_t sharedBytes = sizeof(float) * (size_t)layer.m_TotalNeurons;
    KernelPredict<<<grid, CUDA_BLOCK_SIZE, sharedBytes>>>(
        layer, batchX, results, batchCount);
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
