#include "MLPClassifierNIF.h"
#include "erl_nif.h"

#include <vector>

struct ModelResource
{
    MLPClassifierNIF* model;
};

static ErlNifResourceType* MODEL_RES_TYPE = NULL;

static void model_resource_dtor(ErlNifEnv* env, void* obj)
{
    (void)env;
    ModelResource* resource = (ModelResource*)obj;
    delete resource->model;
    resource->model = NULL;
}

static int load_nif(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    (void)priv_data;
    (void)load_info;

    MODEL_RES_TYPE =
        enif_open_resource_type(env,
                                NULL,
                                "CudaBackpropModel",
                                model_resource_dtor,
                                (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER),
                                NULL);

    return MODEL_RES_TYPE == NULL ? -1 : 0;
}

static bool get_int(ErlNifEnv* env, ERL_NIF_TERM term, int* value)
{
    return enif_get_int(env, term, value) != 0;
}

static bool get_double(ErlNifEnv* env, ERL_NIF_TERM term, double* value)
{
    return enif_get_double(env, term, value) != 0;
}

static bool get_int_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<int>& out)
{
    unsigned int length = 0;
    if (!enif_get_list_length(env, list, &length))
        return false;

    out.clear();
    out.reserve(length);

    ERL_NIF_TERM head;
    ERL_NIF_TERM tail = list;
    for (unsigned int i = 0; i < length; ++i)
    {
        if (!enif_get_list_cell(env, tail, &head, &tail))
            return false;

        int value = 0;
        if (!enif_get_int(env, head, &value))
            return false;

        out.push_back(value);
    }

    return true;
}

static bool get_float_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<float>& out)
{
    unsigned int length = 0;
    if (!enif_get_list_length(env, list, &length))
        return false;

    out.clear();
    out.reserve(length);

    ERL_NIF_TERM head;
    ERL_NIF_TERM tail = list;
    for (unsigned int i = 0; i < length; ++i)
    {
        if (!enif_get_list_cell(env, tail, &head, &tail))
            return false;

        double value = 0.0;
        int intValue = 0;
        if (enif_get_double(env, head, &value))
            out.push_back((float)value);
        else if (enif_get_int(env, head, &intValue))
            out.push_back((float)intValue);
        else
            return false;
    }

    return true;
}

static ERL_NIF_TERM make_float_list(ErlNifEnv* env, const float* values, int count)
{
    ERL_NIF_TERM list = enif_make_list(env, 0);

    for (int i = count - 1; i >= 0; --i)
        list = enif_make_list_cell(env, enif_make_double(env, values[i]), list);

    return list;
}

static ERL_NIF_TERM badarg(ErlNifEnv* env, const char* message)
{
    return enif_make_tuple2(
        env,
        enif_make_atom(env, "error"),
        enif_make_string(env, message, ERL_NIF_LATIN1)
    );
}

static ERL_NIF_TERM new_model_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    if (argc != 3)
        return enif_make_badarg(env);

    std::vector<int> layers;
    std::vector<float> weights;
    std::vector<float> biases;

    if (!get_int_list(env, argv[0], layers))
        return badarg(env, "layers deve ser uma lista de inteiros");
    if (!get_float_list(env, argv[1], weights))
        return badarg(env, "weights deve ser uma lista numerica");
    if (!get_float_list(env, argv[2], biases))
        return badarg(env, "biases deve ser uma lista numerica");
    if (layers.empty())
        return badarg(env, "topologia vazia");

    int expectedWeights = 0;
    int expectedBiases = 0;
    for (size_t i = 1; i < layers.size(); ++i)
    {
        expectedWeights += layers[i - 1] * layers[i];
        expectedBiases += layers[i];
    }

    if ((int)weights.size() != expectedWeights)
        return badarg(env, "weights tem tamanho invalido");
    if ((int)biases.size() != expectedBiases)
        return badarg(env, "biases tem tamanho invalido");

    ModelResource* resource =
        (ModelResource*)enif_alloc_resource(MODEL_RES_TYPE, sizeof(ModelResource));

    resource->model = new MLPClassifierNIF(layers, weights, biases);

    ERL_NIF_TERM term = enif_make_resource(env, resource);
    enif_release_resource(resource);
    return term;
}

static bool get_model(ErlNifEnv* env, ERL_NIF_TERM term, ModelResource** resource)
{
    return enif_get_resource(env, term, MODEL_RES_TYPE, (void**)resource) != 0 &&
           (*resource)->model != NULL;
}

static ERL_NIF_TERM train_batch_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    if (argc != 6)
        return enif_make_badarg(env);

    ModelResource* resource = NULL;
    std::vector<float> trainX;
    std::vector<float> trainY;
    int batchCount = 0;
    int inputSize = 0;
    double learnRate = 0.0;

    if (!get_model(env, argv[0], &resource))
        return badarg(env, "modelo CUDA invalido");
    if (!get_float_list(env, argv[1], trainX))
        return badarg(env, "train_x_flat deve ser uma lista numerica");
    if (!get_float_list(env, argv[2], trainY))
        return badarg(env, "train_y deve ser uma lista numerica");
    if (!get_int(env, argv[3], &batchCount) ||
        !get_int(env, argv[4], &inputSize) ||
        !get_double(env, argv[5], &learnRate))
        return enif_make_badarg(env);

    if ((int)trainX.size() != batchCount * inputSize)
        return badarg(env, "train_x_flat tem tamanho invalido");
    if ((int)trainY.size() != batchCount)
        return badarg(env, "train_y tem tamanho invalido");

    if (!resource->model->TrainBatch(trainX.data(), trainY.data(), batchCount, (float)learnRate))
        return badarg(env, "falha ao executar treino CUDA no NIF");

    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM train_batch_binary_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    if (argc != 6)
        return enif_make_badarg(env);

    ModelResource* resource = NULL;
    ErlNifBinary trainX;
    ErlNifBinary trainY;
    int batchCount = 0;
    int inputSize = 0;
    double learnRate = 0.0;

    if (!get_model(env, argv[0], &resource))
        return badarg(env, "modelo CUDA invalido");
    if (!enif_inspect_binary(env, argv[1], &trainX))
        return badarg(env, "train_x_bin deve ser binary");
    if (!enif_inspect_binary(env, argv[2], &trainY))
        return badarg(env, "train_y_bin deve ser binary");
    if (!get_int(env, argv[3], &batchCount) ||
        !get_int(env, argv[4], &inputSize) ||
        !get_double(env, argv[5], &learnRate))
        return enif_make_badarg(env);

    const size_t expectedX = (size_t)batchCount * (size_t)inputSize * sizeof(float);
    const size_t expectedY = (size_t)batchCount * sizeof(float);

    if (trainX.size != expectedX)
        return badarg(env, "train_x_bin tem tamanho invalido");
    if (trainY.size != expectedY)
        return badarg(env, "train_y_bin tem tamanho invalido");

    if (!resource->model->TrainBatch((float*)trainX.data, (float*)trainY.data, batchCount, (float)learnRate))
        return badarg(env, "falha ao executar treino CUDA no NIF");

    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM predict_batch_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    if (argc != 4)
        return enif_make_badarg(env);

    ModelResource* resource = NULL;
    std::vector<float> features;
    int batchCount = 0;
    int inputSize = 0;

    if (!get_model(env, argv[0], &resource))
        return badarg(env, "modelo CUDA invalido");
    if (!get_float_list(env, argv[1], features))
        return badarg(env, "x_flat deve ser uma lista numerica");
    if (!get_int(env, argv[2], &batchCount) ||
        !get_int(env, argv[3], &inputSize))
        return enif_make_badarg(env);
    if ((int)features.size() != batchCount * inputSize)
        return badarg(env, "x_flat tem tamanho invalido");

    float* pred = resource->model->Predict(features.data(), batchCount);
    if (pred == NULL)
        return badarg(env, "falha ao executar inferencia CUDA no NIF");

    ERL_NIF_TERM predList = make_float_list(env, pred, batchCount);
    delete[] pred;
    return predList;
}

static ERL_NIF_TERM predict_batch_binary_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    if (argc != 4)
        return enif_make_badarg(env);

    ModelResource* resource = NULL;
    ErlNifBinary features;
    int batchCount = 0;
    int inputSize = 0;

    if (!get_model(env, argv[0], &resource))
        return badarg(env, "modelo CUDA invalido");
    if (!enif_inspect_binary(env, argv[1], &features))
        return badarg(env, "x_bin deve ser binary");
    if (!get_int(env, argv[2], &batchCount) ||
        !get_int(env, argv[3], &inputSize))
        return enif_make_badarg(env);

    const size_t expected = (size_t)batchCount * (size_t)inputSize * sizeof(float);
    if (features.size != expected)
        return badarg(env, "x_bin tem tamanho invalido");

    float* pred = resource->model->Predict((float*)features.data, batchCount);
    if (pred == NULL)
        return badarg(env, "falha ao executar inferencia CUDA no NIF");

    ERL_NIF_TERM predList = make_float_list(env, pred, batchCount);
    delete[] pred;
    return predList;
}

static ERL_NIF_TERM last_timings_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    if (argc != 1)
        return enif_make_badarg(env);

    ModelResource* resource = NULL;
    if (!get_model(env, argv[0], &resource))
        return badarg(env, "modelo CUDA invalido");

    long long cpuToGpuUs = 0;
    long long gpuComputeUs = 0;
    long long gpuToCpuUs = 0;
    resource->model->GetLastTimings(&cpuToGpuUs, &gpuComputeUs, &gpuToCpuUs);

    return enif_make_tuple3(
        env,
        enif_make_int64(env, cpuToGpuUs),
        enif_make_int64(env, gpuComputeUs),
        enif_make_int64(env, gpuToCpuUs)
    );
}

static ErlNifFunc nif_funcs[] = {
    {"new_model", 3, new_model_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"train_batch", 6, train_batch_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"train_batch_binary", 6, train_batch_binary_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"predict_batch", 4, predict_batch_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"predict_batch_binary", 4, predict_batch_binary_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"last_timings", 1, last_timings_nif, 0}
};

ERL_NIF_INIT(Elixir.CudaNif, nif_funcs, load_nif, NULL, NULL, NULL)

