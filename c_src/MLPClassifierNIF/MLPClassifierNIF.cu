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
    : m_Initialized(false), m_LayerCount(0), m_cuMLPLayer(NULL)
{
    memset(m_cuWeightsPtr, 0, sizeof(m_cuWeightsPtr));
    memset(m_cuBiasesPtr,  0, sizeof(m_cuBiasesPtr));
    memset(&m_MLPLayer,    0, sizeof(MLPLayer));

    if (cudaSetDevice(0) != cudaSuccess)
    {
        std::fprintf(stderr, "cudaSetDevice falhou. GPU com suporte CUDA disponivel?\n");
        return;
    }

    m_LayerCount               = (int)layers.size();
    m_MLPLayer.m_LayerCount    = m_LayerCount;
    m_MLPLayer.m_TotalNeurons  = 0;
    m_MLPLayer.m_TotalWeights  = 0;
    m_MLPLayer.m_TotalBiases   = 0;

    for (int i = 0; i < m_LayerCount; ++i)
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

    float* hostWeights[MAX_LAYER];
    float* hostBiases[MAX_LAYER];
    memset(hostWeights, 0, sizeof(hostWeights));
    memset(hostBiases,  0, sizeof(hostBiases));

    for (int l = 1; l < m_LayerCount; ++l)
    {
        const int prevSize = layers[l - 1];
        const int currSize = layers[l];
        const int wCount   = prevSize * currSize;
        const int wOff     = m_MLPLayer.m_WeightOffset[l];
        const int bOff     = m_MLPLayer.m_BiasOffset[l];

        hostBiases[l] = new float[currSize];
        hostWeights[l] = new float[wCount];

        for (int n = 0; n < wCount; ++n)
            hostWeights[l][n] = flatWeights[wOff + n];

        for (int n = 0; n < currSize; ++n)
            hostBiases[l][n] = flatBiases[bOff + n];
    }

    cudaError_t cuErr = cudaMalloc((void**)&m_cuMLPLayer, sizeof(MLPLayer));
    if (cuErr != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc (MLPLayer): %s\n", cudaGetErrorString(cuErr));
        goto cleanup;
    }

    for (int l = 1; l < m_LayerCount; ++l)
    {
        cuErr = cudaMalloc((void**)&m_cuBiasesPtr[l],  sizeof(float) * layers[l]);
        if (cuErr != cudaSuccess) goto cleanup;

        cuErr = cudaMemcpy(m_cuBiasesPtr[l], hostBiases[l],
                           sizeof(float) * layers[l], cudaMemcpyHostToDevice);
        if (cuErr != cudaSuccess) goto cleanup;

        cuErr = cudaMalloc((void**)&m_cuWeightsPtr[l],
                           sizeof(float) * layers[l - 1] * layers[l]);
        if (cuErr != cudaSuccess) goto cleanup;

        cuErr = cudaMemcpy(m_cuWeightsPtr[l], hostWeights[l],
                           sizeof(float) * layers[l - 1] * layers[l],
                           cudaMemcpyHostToDevice);
        if (cuErr != cudaSuccess) goto cleanup;
    }

    cuErr = cudaMemcpy(m_cuMLPLayer, &m_MLPLayer, sizeof(MLPLayer), cudaMemcpyHostToDevice);
    if (cuErr != cudaSuccess) goto cleanup;

    for (int l = 1; l < m_LayerCount; ++l)
    {
        cuErr = cudaMemcpy(&m_cuMLPLayer->m_Weights[l], &m_cuWeightsPtr[l],
                           sizeof(float*), cudaMemcpyHostToDevice);
        if (cuErr != cudaSuccess) goto cleanup;

        cuErr = cudaMemcpy(&m_cuMLPLayer->m_Biases[l], &m_cuBiasesPtr[l],
                           sizeof(float*), cudaMemcpyHostToDevice);
        if (cuErr != cudaSuccess) goto cleanup;
    }

    m_Initialized = true;

    {
        const char* dbgEnv = std::getenv("BACKPROP_DEBUG");
        if (dbgEnv != NULL && dbgEnv[0] == '1')
        {
            std::printf("[DEBUG] Pesos iniciais\n");
            for (int l = 1; l < m_LayerCount; ++l)
            {
                const int take = std::min(4, layers[l - 1] * layers[l]);
                std::printf("[DEBUG] Layer %d:", l);
                for (int k = 0; k < take; ++k)
                    std::printf(" %.6f", hostWeights[l][k]);
                std::printf("\n");
            }
        }
    }

cleanup:
    for (int l = 1; l < m_LayerCount; ++l)
    {
        delete[] hostWeights[l];
        delete[] hostBiases[l];
    }

    if (!m_Initialized)
        std::fprintf(stderr, "Falha na inicializacao: %s\n", cudaGetErrorString(cuErr));
}

MLPClassifierNIF::~MLPClassifierNIF()
{
    if (m_Initialized)
    {
        if (m_cuMLPLayer)
            cudaFree(m_cuMLPLayer);

        for (int l = 1; l < MAX_LAYER; ++l)
        {
            if (m_cuBiasesPtr[l])  cudaFree(m_cuBiasesPtr[l]);
            if (m_cuWeightsPtr[l]) cudaFree(m_cuWeightsPtr[l]);
        }

        if (cudaDeviceReset() != cudaSuccess)
            std::fprintf(stderr, "cudaDeviceReset falhou.\n");
    }
}

void MLPClassifierNIF::TrainBatch(float* trainx, float* trainy,
                                  int batchCount, float learnRate)
{
    if (!m_Initialized || batchCount <= 0)
        return;

    const int inputSize    = m_MLPLayer.m_Layers[0];
    const int totalNeurons = m_MLPLayer.m_TotalNeurons;
    const int totalWeights = m_MLPLayer.m_TotalWeights;
    const int totalBiases  = m_MLPLayer.m_TotalBiases;

    float *cu_gradW  = NULL, *cu_gradB  = NULL;
    float *cu_act    = NULL, *cu_delta  = NULL;
    float *cu_batchX = NULL, *cu_batchY = NULL;

    cudaMalloc(&cu_gradW,  sizeof(float) * totalWeights);
    cudaMalloc(&cu_gradB,  sizeof(float) * totalBiases);
    cudaMalloc(&cu_act,    sizeof(float) * totalNeurons * batchCount);
    cudaMalloc(&cu_delta,  sizeof(float) * totalNeurons * batchCount);
    cudaMalloc(&cu_batchX, sizeof(float) * inputSize    * batchCount);
    cudaMalloc(&cu_batchY, sizeof(float) * batchCount);

    float* hBatchX = new float[(size_t)inputSize * batchCount];
    float* hBatchY = new float[batchCount];

    const char* dbgEnv = std::getenv("BACKPROP_DEBUG");
    const bool  debug  = (dbgEnv != NULL && dbgEnv[0] == '1');
    const int debugCount = DebugValueCount();
    float dbgBuf[4];

    if (debug)
        PrintDebugSnapshot(0, m_MLPLayer, m_cuWeightsPtr, m_cuBiasesPtr, NULL, NULL, debugCount);

    cudaMemset(cu_gradW, 0, sizeof(float) * totalWeights);
    cudaMemset(cu_gradB, 0, sizeof(float) * totalBiases);

    for (int f = 0; f < inputSize; ++f)
        for (int s = 0; s < batchCount; ++s)
            hBatchX[f * batchCount + s] = trainx[s * inputSize + f];

    for (int s = 0; s < batchCount; ++s)
        hBatchY[s] = trainy[s];

    cudaMemcpy(cu_batchX, hBatchX, sizeof(float) * inputSize * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_batchY, hBatchY, sizeof(float) * batchCount,             cudaMemcpyHostToDevice);

    LaunchFitKernel(m_cuMLPLayer, cu_batchX, cu_batchY,
                    batchCount, cu_act, cu_delta, cu_gradW, cu_gradB);
    cudaDeviceSynchronize();

    for (int l = 1; l < m_LayerCount; ++l)
    {
        const int wCount = m_MLPLayer.m_Layers[l - 1] * m_MLPLayer.m_Layers[l];
        const int bCount = m_MLPLayer.m_Layers[l];
        const int wOff   = m_MLPLayer.m_WeightOffset[l];
        const int bOff   = m_MLPLayer.m_BiasOffset[l];

        LaunchUpdateWeightsKernel(
            m_cuWeightsPtr[l], m_cuBiasesPtr[l],
            cu_gradW + wOff,   cu_gradB + bOff,
            wCount, bCount, learnRate, batchCount);
    }
    cudaDeviceSynchronize();

    if (debug)
    {
        PrintDebugSnapshot(1, m_MLPLayer, m_cuWeightsPtr, m_cuBiasesPtr, cu_gradW, cu_gradB, debugCount);
        std::printf("[DEBUG] Batch treinado\n");
        for (int l = 1; l < m_LayerCount; ++l)
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

    delete[] hBatchX;
    delete[] hBatchY;

    cudaFree(cu_gradW);
    cudaFree(cu_gradB);
    cudaFree(cu_act);
    cudaFree(cu_delta);
    cudaFree(cu_batchX);
    cudaFree(cu_batchY);
}

float* MLPClassifierNIF::Predict(float* testx, int testCount)
{
    if (!m_Initialized)
        return NULL;

    const int inputSize    = m_MLPLayer.m_Layers[0];
    const int outputSize   = m_MLPLayer.m_Layers[m_LayerCount - 1];
    const int totalNeurons = m_MLPLayer.m_TotalNeurons;

    float* results = new float[(size_t)testCount * outputSize];

    float *cu_act     = NULL;
    float *cu_batchX  = NULL;
    float *cu_results = NULL;

    cudaMalloc(&cu_act,     sizeof(float) * totalNeurons * testCount);
    cudaMalloc(&cu_batchX,  sizeof(float) * inputSize    * testCount);
    cudaMalloc(&cu_results, sizeof(float) * outputSize   * testCount);

    float* hBatchX  = new float[(size_t)inputSize  * testCount];
    float* hResults = new float[(size_t)outputSize * testCount];

    for (int f = 0; f < inputSize; ++f)
        for (int s = 0; s < testCount; ++s)
            hBatchX[f * testCount + s] = testx[s * inputSize + f];

    cudaMemcpy(cu_batchX, hBatchX, sizeof(float) * inputSize * testCount, cudaMemcpyHostToDevice);

    LaunchPredictKernel(m_cuMLPLayer, cu_batchX, cu_results, testCount, cu_act);
    cudaDeviceSynchronize();

    cudaMemcpy(hResults, cu_results, sizeof(float) * outputSize * testCount, cudaMemcpyDeviceToHost);
    for (int s = 0; s < testCount; ++s)
        for (int j = 0; j < outputSize; ++j)
            results[s * outputSize + j] = hResults[j * testCount + s];

    delete[] hBatchX;
    delete[] hResults;

    cudaFree(cu_act);
    cudaFree(cu_batchX);
    cudaFree(cu_results);

    return results;
}
