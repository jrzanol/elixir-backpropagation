#include "MLPClassifierNIF.h"
#include "MLPClassifierNIFDevice.h"

#include "cuda_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int DebugValueCount()
{
    const char* value = std::getenv("BACKPROP_DEBUG_VALUES");
    if (value == NULL)
        return 8;

    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 8;
}

static void PrintDebugValues(const char* name, const std::vector<float>& values)
{
    std::printf(" %s=[", name);
    for (size_t i = 0; i < values.size(); ++i)
        std::printf("%s%.9f", i == 0 ? "" : ",", values[i]);
    std::printf("]");
}

static std::vector<float> CollectContiguousSamples(float* devicePtr, int totalCount, int sampleCount)
{
    if (devicePtr == NULL)
        return std::vector<float>();

    const int take = std::min(totalCount, sampleCount);
    std::vector<float> values(take);

    if (take > 0)
        cudaMemcpy(values.data(), devicePtr, sizeof(float) * take, cudaMemcpyDeviceToHost);

    return values;
}

static std::vector<float> CollectLayerSamples(float* const* devicePtrs,
                                              const MLPLayer& layer,
                                              bool weights,
                                              int sampleCount)
{
    std::vector<float> values;
    values.reserve(sampleCount);

    for (int l = 1; l < layer.m_LayerCount && (int)values.size() < sampleCount; ++l)
    {
        const int count = weights
            ? layer.m_Layers[l - 1] * layer.m_Layers[l]
            : layer.m_Layers[l];

        const int remaining = sampleCount - (int)values.size();
        const int take = std::min(count, remaining);
        if (take <= 0 || devicePtrs[l] == NULL)
            continue;

        const size_t oldSize = values.size();
        values.resize(oldSize + take);
        cudaMemcpy(values.data() + oldSize, devicePtrs[l], sizeof(float) * take, cudaMemcpyDeviceToHost);
    }

    return values;
}

static void PrintDebugSnapshot(int epoch,
                               const MLPLayer& layer,
                               float* const* weights,
                               float* const* biases,
                               float* gradW,
                               float* gradB,
                               int sampleCount)
{
    std::printf("[DEBUG_SNAPSHOT] impl=cuda epoch=%d", epoch);
    PrintDebugValues("weights", CollectLayerSamples(weights, layer, true, sampleCount));
    PrintDebugValues("biases",  CollectLayerSamples(biases,  layer, false, sampleCount));
    PrintDebugValues("grad_w",  CollectContiguousSamples(gradW, layer.m_TotalWeights, sampleCount));
    PrintDebugValues("grad_b",  CollectContiguousSamples(gradB, layer.m_TotalBiases, sampleCount));
    std::printf("\n");
}

MLPClassifierNIF::MLPClassifierNIF(const std::vector<int>& layers,
                                   const std::vector<float>& flatWeights,
                                   const std::vector<float>& flatBiases)
    : m_Initialized(false),
      m_BatchCapacity(0),
      m_cuMLPLayer(NULL),
      m_cuGradW(NULL),
      m_cuGradB(NULL),
      m_cuAct(NULL),
      m_cuDelta(NULL),
      m_cuBatchX(NULL),
      m_cuBatchY(NULL),
      m_cuResults(NULL)
{
    memset(m_cuWeightsPtr, 0, sizeof(m_cuWeightsPtr));
    memset(m_cuBiasesPtr,  0, sizeof(m_cuBiasesPtr));
    memset(&m_MLPLayer,    0, sizeof(MLPLayer));

    if (cudaSetDevice(0) != cudaSuccess)
    {
        std::fprintf(stderr, "cudaSetDevice falhou. GPU com suporte CUDA disponivel?\n");
        return;
    }

    if (layers.size() < 2 || layers.size() > MAX_LAYER)
    {
        std::fprintf(stderr, "Topologia deve possuir entre 2 e %d camadas.\n", MAX_LAYER);
        return;
    }

    m_MLPLayer.m_LayerCount    = (int)layers.size();
    m_MLPLayer.m_TotalNeurons  = 0;
    m_MLPLayer.m_TotalWeights  = 0;
    m_MLPLayer.m_TotalBiases   = 0;

    for (int i = 0; i < m_MLPLayer.m_LayerCount; ++i)
    {
        m_MLPLayer.m_Layers[i]       = layers[i];
        m_MLPLayer.m_NeuronOffset[i] = m_MLPLayer.m_TotalNeurons;
        m_MLPLayer.m_TotalNeurons   += layers[i];

        if (i > 0)
        {
            m_MLPLayer.m_WeightOffset[i] = m_MLPLayer.m_TotalWeights;
            m_MLPLayer.m_TotalWeights   += layers[i - 1] * layers[i];
            m_MLPLayer.m_BiasOffset[i]   = m_MLPLayer.m_TotalBiases;
            m_MLPLayer.m_TotalBiases    += layers[i];
        }
    }

    cudaError_t cuErr = cudaMalloc((void**)&m_cuGradW, sizeof(float) * m_MLPLayer.m_TotalWeights);
    if (cuErr != cudaSuccess) goto cleanup;

    cuErr = cudaMalloc((void**)&m_cuGradB, sizeof(float) * m_MLPLayer.m_TotalBiases);
    if (cuErr != cudaSuccess) goto cleanup;

    cuErr = cudaMalloc((void**)&m_cuMLPLayer, sizeof(MLPLayer));
    if (cuErr != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc (MLPLayer): %s\n", cudaGetErrorString(cuErr));
        goto cleanup;
    }

    for (int l = 1; l < m_MLPLayer.m_LayerCount; ++l)
    {
        const int weightOffset = m_MLPLayer.m_WeightOffset[l];
        const int biasOffset = m_MLPLayer.m_BiasOffset[l];

        cuErr = cudaMalloc((void**)&m_cuBiasesPtr[l],  sizeof(float) * layers[l]);
        if (cuErr != cudaSuccess) goto cleanup;

        cuErr = cudaMemcpy(m_cuBiasesPtr[l], flatBiases.data() + biasOffset,
                           sizeof(float) * layers[l], cudaMemcpyHostToDevice);
        if (cuErr != cudaSuccess) goto cleanup;

        cuErr = cudaMalloc((void**)&m_cuWeightsPtr[l],
                           sizeof(float) * layers[l - 1] * layers[l]);
        if (cuErr != cudaSuccess) goto cleanup;

        cuErr = cudaMemcpy(m_cuWeightsPtr[l], flatWeights.data() + weightOffset,
                           sizeof(float) * layers[l - 1] * layers[l],
                           cudaMemcpyHostToDevice);
        if (cuErr != cudaSuccess) goto cleanup;

        m_MLPLayer.m_Weights[l] = m_cuWeightsPtr[l];
        m_MLPLayer.m_Biases[l] = m_cuBiasesPtr[l];
    }

    cuErr = cudaMemcpy(m_cuMLPLayer, &m_MLPLayer, sizeof(MLPLayer), cudaMemcpyHostToDevice);
    if (cuErr != cudaSuccess) goto cleanup;

    m_Initialized = true;

    {
        const char* dbgEnv = std::getenv("BACKPROP_DEBUG");
        if (dbgEnv != NULL && dbgEnv[0] == '1')
        {
            std::printf("[DEBUG] Pesos iniciais\n");
            for (int l = 1; l < m_MLPLayer.m_LayerCount; ++l)
            {
                const int take = std::min(4, layers[l - 1] * layers[l]);
                std::printf("[DEBUG] Layer %d:", l);
                for (int k = 0; k < take; ++k)
                    std::printf(" %.6f", flatWeights[m_MLPLayer.m_WeightOffset[l] + k]);
                std::printf("\n");
            }
        }
    }

cleanup:
    if (!m_Initialized)
        std::fprintf(stderr, "Falha na inicializacao: %s\n", cudaGetErrorString(cuErr));
}

MLPClassifierNIF::~MLPClassifierNIF()
{
    if (m_cuMLPLayer)
        cudaFree(m_cuMLPLayer);

    for (int l = 1; l < MAX_LAYER; ++l)
    {
        if (m_cuBiasesPtr[l])  cudaFree(m_cuBiasesPtr[l]);
        if (m_cuWeightsPtr[l]) cudaFree(m_cuWeightsPtr[l]);
    }

    if (m_cuGradW)   cudaFree(m_cuGradW);
    if (m_cuGradB)   cudaFree(m_cuGradB);
    if (m_cuAct)     cudaFree(m_cuAct);
    if (m_cuDelta)   cudaFree(m_cuDelta);
    if (m_cuBatchX)  cudaFree(m_cuBatchX);
    if (m_cuBatchY)  cudaFree(m_cuBatchY);
    if (m_cuResults) cudaFree(m_cuResults);
}

bool MLPClassifierNIF::EnsureBatchCapacity(int batchCount)
{
    if (batchCount <= m_BatchCapacity)
        return true;

    if (m_cuAct)     cudaFree(m_cuAct);
    if (m_cuDelta)   cudaFree(m_cuDelta);
    if (m_cuBatchX)  cudaFree(m_cuBatchX);
    if (m_cuBatchY)  cudaFree(m_cuBatchY);
    if (m_cuResults) cudaFree(m_cuResults);

    m_cuAct = m_cuDelta = m_cuBatchX = m_cuBatchY = m_cuResults = NULL;
    m_BatchCapacity = 0;

    const int inputSize = m_MLPLayer.m_Layers[0];
    const int outputSize = m_MLPLayer.m_Layers[m_MLPLayer.m_LayerCount - 1];
    const int totalNeurons = m_MLPLayer.m_TotalNeurons;

    if (cudaMalloc(&m_cuAct, sizeof(float) * totalNeurons * batchCount) != cudaSuccess ||
        cudaMalloc(&m_cuDelta, sizeof(float) * totalNeurons * batchCount) != cudaSuccess ||
        cudaMalloc(&m_cuBatchX, sizeof(float) * inputSize * batchCount) != cudaSuccess ||
        cudaMalloc(&m_cuBatchY, sizeof(float) * batchCount) != cudaSuccess ||
        cudaMalloc(&m_cuResults, sizeof(float) * outputSize * batchCount) != cudaSuccess)
    {
        std::fprintf(stderr, "Falha ao alocar buffers CUDA para %d amostras.\n", batchCount);
        if (m_cuAct)     cudaFree(m_cuAct);
        if (m_cuDelta)   cudaFree(m_cuDelta);
        if (m_cuBatchX)  cudaFree(m_cuBatchX);
        if (m_cuBatchY)  cudaFree(m_cuBatchY);
        if (m_cuResults) cudaFree(m_cuResults);
        m_cuAct = m_cuDelta = m_cuBatchX = m_cuBatchY = m_cuResults = NULL;
        return false;
    }

    m_BatchCapacity = batchCount;
    return true;
}

bool MLPClassifierNIF::TrainBatch(float* trainx, float* trainy,
                                  int batchCount, float learnRate)
{
    if (!m_Initialized || batchCount <= 0 || !EnsureBatchCapacity(batchCount))
        return false;

    const int inputSize    = m_MLPLayer.m_Layers[0];
    const int totalWeights = m_MLPLayer.m_TotalWeights;
    const int totalBiases  = m_MLPLayer.m_TotalBiases;

    const char* dbgEnv = std::getenv("BACKPROP_DEBUG");
    const bool  debug  = (dbgEnv != NULL && dbgEnv[0] == '1');
    const int debugCount = DebugValueCount();
    float dbgBuf[4];

    if (debug)
        PrintDebugSnapshot(0, m_MLPLayer, m_cuWeightsPtr, m_cuBiasesPtr, NULL, NULL, debugCount);

    cudaMemset(m_cuGradW, 0, sizeof(float) * totalWeights);
    cudaMemset(m_cuGradB, 0, sizeof(float) * totalBiases);

    cudaMemcpy(m_cuBatchX, trainx, sizeof(float) * inputSize * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(m_cuBatchY, trainy, sizeof(float) * batchCount, cudaMemcpyHostToDevice);

    LaunchFitKernel(m_cuMLPLayer, m_cuBatchX, m_cuBatchY,
                    batchCount, m_cuAct, m_cuDelta, m_cuGradW, m_cuGradB);

    for (int l = 1; l < m_MLPLayer.m_LayerCount; ++l)
    {
        const int wCount = m_MLPLayer.m_Layers[l - 1] * m_MLPLayer.m_Layers[l];
        const int bCount = m_MLPLayer.m_Layers[l];
        const int wOff   = m_MLPLayer.m_WeightOffset[l];
        const int bOff   = m_MLPLayer.m_BiasOffset[l];

        LaunchUpdateWeightsKernel(
            m_cuWeightsPtr[l], m_cuBiasesPtr[l],
            m_cuGradW + wOff,  m_cuGradB + bOff,
            wCount, bCount, learnRate, batchCount);
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;

    if (debug)
    {
        PrintDebugSnapshot(1, m_MLPLayer, m_cuWeightsPtr, m_cuBiasesPtr, m_cuGradW, m_cuGradB, debugCount);
        std::printf("[DEBUG] Batch treinado\n");
        for (int l = 1; l < m_MLPLayer.m_LayerCount; ++l)
        {
            const int wCount = m_MLPLayer.m_Layers[l - 1] * m_MLPLayer.m_Layers[l];
            const int take   = wCount < 4 ? wCount : 4;
            cudaMemcpy(dbgBuf, m_cuWeightsPtr[l], sizeof(float) * take, cudaMemcpyDeviceToHost);
            std::printf("[DEBUG] Layer %d:", l);
            for (int k = 0; k < take; ++k)
                std::printf(" %.6f", dbgBuf[k]);
            std::printf("\n");
        }
    }

    return true;
}

float* MLPClassifierNIF::Predict(float* testx, int testCount)
{
    if (!m_Initialized || testCount <= 0 || !EnsureBatchCapacity(testCount))
        return NULL;

    const int inputSize    = m_MLPLayer.m_Layers[0];
    const int outputSize   = m_MLPLayer.m_Layers[m_MLPLayer.m_LayerCount - 1];
    float* results = new float[(size_t)testCount * outputSize];

    cudaMemcpy(m_cuBatchX, testx, sizeof(float) * inputSize * testCount, cudaMemcpyHostToDevice);

    LaunchPredictKernel(m_cuMLPLayer, m_cuBatchX, m_cuResults, testCount, m_cuAct);

    if (cudaMemcpy(results, m_cuResults, sizeof(float) * outputSize * testCount, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        delete[] results;
        return NULL;
    }

    return results;
}
