defmodule Model do
  def new(%{layers: layers, weights: weights, biases: biases}) do
    %{ref: CudaNif.new_model(layers, List.flatten(weights), List.flatten(biases))}
  end

  def load_train_batch(path), do: Dataset.load_flat_batch(path)
  def load_predict_batch(path), do: Dataset.load_flat_batch(path)

  def train_batch(model, batch, n_features, learn_rate) do
    features = Dataset.flatten_features(batch)
    labels = Dataset.flatten_labels(batch)

    Profiler.runtime(:train_batch, fn ->
      :ok = CudaNif.train_batch(model.ref, features, labels, batch.count, n_features, learn_rate)
      record_cuda_timings(model.ref, :train)
    end)

    model
  end

  def predict_batch(model, batch, n_features) do
    predict_probabilities(model, batch, n_features)
    |> Enum.map(&prob_to_class/1)
  end

  def predict_probabilities(model, batch, n_features) do
    features = Dataset.flatten_features(batch)

    Profiler.runtime(:predict_batch, fn ->
      probabilities = CudaNif.predict_batch(model.ref, features, batch.count, n_features)
      record_cuda_timings(model.ref, :predict)
      probabilities
    end)
  end

  def train_epoch(model, batch_paths, n_features, learn_rate) do
    reduce_prefetched_batches(batch_paths, model, :load_train_batch, fn batch, model ->
      Profiler.runtime(:train_batch, fn ->
        :ok =
          CudaNif.train_batch_binary(
            model.ref,
            batch.features_bin,
            batch.labels_bin,
            batch.count,
            n_features,
            learn_rate
          )

        record_cuda_timings(model.ref, :train)
      end)

      model
    end)
  end

  def evaluate_paths(model, batch_paths, n_features) do
    reduce_prefetched_batches(batch_paths, Metrics.new(), :load_predict_batch, fn batch,
                                                                                  metrics ->
      probabilities = predict_binary_probabilities(model, batch, n_features)
      predictions = Enum.map(probabilities, &prob_to_class/1)
      Metrics.update(metrics, predictions, batch.labels)
    end)
  end

  def predict_binary_probabilities(model, batch, n_features) do
    Profiler.runtime(:predict_batch, fn ->
      probabilities =
        CudaNif.predict_batch_binary(model.ref, batch.features_bin, batch.count, n_features)

      record_cuda_timings(model.ref, :predict)
      probabilities
    end)
  end

  defp reduce_prefetched_batches([], accumulator, _load_event, _fun), do: accumulator

  defp reduce_prefetched_batches([first_path | remaining_paths], accumulator, load_event, fun) do
    first_batch = timed_read_raw_batch!(first_path)

    {accumulator, pending_task} =
      Enum.reduce(remaining_paths, {accumulator, nil}, fn path, {accumulator, pending_task} ->
        next_task = Task.async(fn -> timed_read_raw_batch!(path) end)
        timed_batch = if pending_task, do: Task.await(pending_task, :infinity), else: first_batch
        {fun.(record_load(timed_batch, load_event), accumulator), next_task}
      end)

    timed_batch = if pending_task, do: Task.await(pending_task, :infinity), else: first_batch
    fun.(record_load(timed_batch, load_event), accumulator)
  end

  defp timed_read_raw_batch!(path) do
    {us, batch} = :timer.tc(fn -> read_raw_batch!(path) end)
    {batch, us}
  end

  defp record_load({batch, us}, event) do
    Profiler.record(event, us)
    batch
  end

  defp read_raw_batch!(path) do
    <<"BPBATCH1", count::unsigned-little-32, n_features::unsigned-little-32, payload::binary>> =
      File.read!(path)

    feature_bytes = count * n_features * 4
    label_bytes = count * 4
    <<features_bin::binary-size(feature_bytes), labels_bin::binary-size(label_bytes)>> = payload

    %{
      features_bin: features_bin,
      labels_bin: labels_bin,
      labels: for(<<value::float-little-32 <- labels_bin>>, do: value),
      count: count
    }
  end

  defp record_cuda_timings(model_ref, operation) do
    if Profiler.enabled?() do
      {cpu_to_gpu_us, gpu_compute_us, gpu_to_cpu_us} = CudaNif.last_timings(model_ref)

      case operation do
        :train ->
          Profiler.record(:train_cpu_gpu_transfer, cpu_to_gpu_us)
          Profiler.record(:train_gpu_compute, gpu_compute_us)

        :predict ->
          Profiler.record(:predict_cpu_gpu_transfer, cpu_to_gpu_us)
          Profiler.record(:predict_gpu_compute, gpu_compute_us)
          Profiler.record(:predict_gpu_cpu_transfer, gpu_to_cpu_us)
      end
    end
  end

  defp prob_to_class(prob), do: if(prob >= 0.5, do: 1, else: 0)
end
