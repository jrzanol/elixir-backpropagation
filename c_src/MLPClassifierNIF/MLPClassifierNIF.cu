#include "MLPClassifierNIF.h"
#include "MLPClassifierNIFDevice.h"

#include "cuda_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool CudaOk(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess)
        return true;

    std::fprintf(stderr, "CUDA %s falhou: %s\n", operation, cudaGetErrorString(error));
    return false;
}

static long long ElapsedUs(std::chrono::steady_clock::time_point started)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();
}

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

static void PrintDebugSnapshot(int epoch,
                               const MLPLayer& layer,
                               float* weights,
                               float* biases,
                               float* gradW,
                               float* gradB,
                               int sampleCount)
{
    std::printf("[DEBUG_SNAPSHOT] impl=cuda epoch=%d", epoch);
    PrintDebugValues("weights", CollectContiguousSamples(weights, layer.m_TotalWeights, sampleCount));
    PrintDebugValues("biases",  CollectContiguousSamples(biases, layer.m_TotalBiases, sampleCount));
    PrintDebugValues("grad_w",  CollectContiguousSamples(gradW, layer.m_TotalWeights, sampleCount));
    PrintDebugValues("grad_b",  CollectContiguousSamples(gradB, layer.m_TotalBiases, sampleCount));
    std::printf("\n");
}

MLPClassifierNIF::MLPClassifierNIF(const std::vector<int>& layers,
                                   const std::vector<float>& flatWeights,
                                   const std::vector<float>& flatBiases)
    : m_Initialized(false),
      m_DebugSnapshotPrinted(false),
      m_BatchCapacity(0),
      m_cuWeights(NULL),
      m_cuBiases(NULL),
      m_cuGradW(NULL),
      m_cuGradB(NULL),
      m_cuBatchX(NULL),
      m_cuBatchY(NULL),
      m_cuResults(NULL),
      m_LastCpuToGpuUs(0),
      m_LastGpuComputeUs(0),
      m_LastGpuToCpuUs(0)
{
    memset(&m_MLPLayer, 0, sizeof(MLPLayer));

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

    cudaError_t cuErr = cudaMalloc((void**)&m_cuWeights,
                                   sizeof(float) * m_MLPLayer.m_TotalWeights);
    if (cuErr != cudaSuccess) goto cleanup;

    cuErr = cudaMemcpy(m_cuWeights, flatWeights.data(),
                       sizeof(float) * m_MLPLayer.m_TotalWeights,
                       cudaMemcpyHostToDevice);
    if (cuErr != cudaSuccess) goto cleanup;

    cuErr = cudaMalloc((void**)&m_cuBiases, sizeof(float) * m_MLPLayer.m_TotalBiases);
    if (cuErr != cudaSuccess) goto cleanup;

    cuErr = cudaMemcpy(m_cuBiases, flatBiases.data(),
                       sizeof(float) * m_MLPLayer.m_TotalBiases,
                       cudaMemcpyHostToDevice);
    if (cuErr != cudaSuccess) goto cleanup;

    cuErr = cudaMalloc((void**)&m_cuGradW, sizeof(float) * m_MLPLayer.m_TotalWeights);
    if (cuErr != cudaSuccess) goto cleanup;

    cuErr = cudaMalloc((void**)&m_cuGradB, sizeof(float) * m_MLPLayer.m_TotalBiases);
    if (cuErr != cudaSuccess) goto cleanup;

    m_MLPLayer.m_Weights = m_cuWeights;
    m_MLPLayer.m_Biases = m_cuBiases;

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
    if (m_cuWeights) cudaFree(m_cuWeights);
    if (m_cuBiases)  cudaFree(m_cuBiases);
    if (m_cuGradW)   cudaFree(m_cuGradW);
    if (m_cuGradB)   cudaFree(m_cuGradB);
    if (m_cuBatchX)  cudaFree(m_cuBatchX);
    if (m_cuBatchY)  cudaFree(m_cuBatchY);
    if (m_cuResults) cudaFree(m_cuResults);
}

bool MLPClassifierNIF::EnsureBatchCapacity(int batchCount)
{
    if (batchCount <= m_BatchCapacity)
        return true;

    if (m_cuBatchX)  cudaFree(m_cuBatchX);
    if (m_cuBatchY)  cudaFree(m_cuBatchY);
    if (m_cuResults) cudaFree(m_cuResults);

    m_cuBatchX = m_cuBatchY = m_cuResults = NULL;
    m_BatchCapacity = 0;

    const int inputSize = m_MLPLayer.m_Layers[0];
    const int outputSize = m_MLPLayer.m_Layers[m_MLPLayer.m_LayerCount - 1];
    if (cudaMalloc(&m_cuBatchX, sizeof(float) * inputSize * batchCount) != cudaSuccess ||
        cudaMalloc(&m_cuBatchY, sizeof(float) * batchCount) != cudaSuccess ||
        cudaMalloc(&m_cuResults, sizeof(float) * outputSize * batchCount) != cudaSuccess)
    {
        std::fprintf(stderr, "Falha ao alocar buffers CUDA para %d amostras.\n", batchCount);
        if (m_cuBatchX)  cudaFree(m_cuBatchX);
        if (m_cuBatchY)  cudaFree(m_cuBatchY);
        if (m_cuResults) cudaFree(m_cuResults);
        m_cuBatchX = m_cuBatchY = m_cuResults = NULL;
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
    m_LastCpuToGpuUs = 0;
    m_LastGpuComputeUs = 0;
    m_LastGpuToCpuUs = 0;

    const char* dbgEnv = std::getenv("BACKPROP_DEBUG");
    const bool  debug  = (dbgEnv != NULL && dbgEnv[0] == '1' && !m_DebugSnapshotPrinted);
    const int debugCount = DebugValueCount();
    float dbgBuf[4];

    if (debug)
        PrintDebugSnapshot(0, m_MLPLayer, m_cuWeights, m_cuBiases, NULL, NULL, debugCount);

    const auto transferStarted = std::chrono::steady_clock::now();
    if (!CudaOk(cudaMemcpy(m_cuBatchX, trainx, sizeof(float) * inputSize * batchCount,
                           cudaMemcpyHostToDevice), "copia batch X para GPU") ||
        !CudaOk(cudaMemcpy(m_cuBatchY, trainy, sizeof(float) * batchCount,
                           cudaMemcpyHostToDevice), "copia batch Y para GPU"))
        return false;
    m_LastCpuToGpuUs = ElapsedUs(transferStarted);

    const auto computeStarted = std::chrono::steady_clock::now();
    if (!CudaOk(cudaMemset(m_cuGradW, 0, sizeof(float) * totalWeights), "cudaMemset gradW") ||
        !CudaOk(cudaMemset(m_cuGradB, 0, sizeof(float) * totalBiases), "cudaMemset gradB"))
        return false;

    LaunchFitKernel(m_MLPLayer, m_cuBatchX, m_cuBatchY,
                    batchCount, m_cuGradW, m_cuGradB);
    if (!CudaOk(cudaGetLastError(), "lancamento KernelFit"))
        return false;

    LaunchUpdateWeightsKernel(
        m_cuWeights, m_cuBiases, m_cuGradW, m_cuGradB,
        totalWeights, totalBiases, learnRate, batchCount);
    if (!CudaOk(cudaGetLastError(), "lancamento KernelUpdateWeights"))
        return false;

    if (!CudaOk(cudaDeviceSynchronize(), "sincronizacao do treino"))
        return false;
    m_LastGpuComputeUs = ElapsedUs(computeStarted);

    if (debug)
    {
        PrintDebugSnapshot(1, m_MLPLayer, m_cuWeights, m_cuBiases, m_cuGradW, m_cuGradB, debugCount);
        std::printf("[DEBUG] Batch treinado\n");
        for (int l = 1; l < m_MLPLayer.m_LayerCount; ++l)
        {
            const int wCount = m_MLPLayer.m_Layers[l - 1] * m_MLPLayer.m_Layers[l];
            const int take   = wCount < 4 ? wCount : 4;
            cudaMemcpy(dbgBuf, m_cuWeights + m_MLPLayer.m_WeightOffset[l],
                       sizeof(float) * take, cudaMemcpyDeviceToHost);
            std::printf("[DEBUG] Layer %d:", l);
            for (int k = 0; k < take; ++k)
                std::printf(" %.6f", dbgBuf[k]);
            std::printf("\n");
        }
        m_DebugSnapshotPrinted = true;
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
    m_LastCpuToGpuUs = 0;
    m_LastGpuComputeUs = 0;
    m_LastGpuToCpuUs = 0;

    const auto transferStarted = std::chrono::steady_clock::now();
    if (!CudaOk(cudaMemcpy(m_cuBatchX, testx, sizeof(float) * inputSize * testCount,
                           cudaMemcpyHostToDevice), "copia batch de predicao para GPU"))
    {
        delete[] results;
        return NULL;
    }
    m_LastCpuToGpuUs = ElapsedUs(transferStarted);

    const auto computeStarted = std::chrono::steady_clock::now();
    LaunchPredictKernel(m_MLPLayer, m_cuBatchX, m_cuResults, testCount);
    if (!CudaOk(cudaGetLastError(), "lancamento KernelPredict") ||
        !CudaOk(cudaDeviceSynchronize(), "sincronizacao da predicao"))
    {
        delete[] results;
        return NULL;
    }
    m_LastGpuComputeUs = ElapsedUs(computeStarted);

    const auto copyBackStarted = std::chrono::steady_clock::now();
    if (!CudaOk(cudaMemcpy(results, m_cuResults, sizeof(float) * outputSize * testCount,
                           cudaMemcpyDeviceToHost), "copia predicoes para CPU"))
    {
        delete[] results;
        return NULL;
    }
    m_LastGpuToCpuUs = ElapsedUs(copyBackStarted);

    return results;
}

void MLPClassifierNIF::GetLastTimings(long long* cpuToGpuUs,
                                      long long* gpuComputeUs,
                                      long long* gpuToCpuUs) const
{
    *cpuToGpuUs = m_LastCpuToGpuUs;
    *gpuComputeUs = m_LastGpuComputeUs;
    *gpuToCpuUs = m_LastGpuToCpuUs;
}
