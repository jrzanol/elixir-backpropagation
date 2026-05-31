#pragma once
#include <vector>

const int MAX_LAYER = 8;

struct MLPLayer
{
    int m_LayerCount;
    int m_Layers[MAX_LAYER];

    // Mapeamento de (camada, neurônio): O(1) dentro dos kernels.
    int m_NeuronOffset[MAX_LAYER];
    int m_WeightOffset[MAX_LAYER];
    int m_BiasOffset[MAX_LAYER];

    int m_TotalNeurons;
    int m_TotalWeights;
    int m_TotalBiases;

    float* m_Weights[MAX_LAYER];
    float* m_Biases[MAX_LAYER];
};

class MLPClassifierNIF
{
public:
    MLPClassifierNIF(const std::vector<int>& layers,
                     const std::vector<float>& flatWeights,
                     const std::vector<float>& flatBiases);
    ~MLPClassifierNIF();

    void TrainBatch(float* trainx, float* trainy, int batchCount, float learnRate);
    float* Predict(float* testx, int testCount);

private:
    bool m_Initialized;
    int  m_LayerCount;

    MLPLayer  m_MLPLayer;
    MLPLayer* m_cuMLPLayer;

    float* m_cuWeightsPtr[MAX_LAYER];
    float* m_cuBiasesPtr[MAX_LAYER];
};
