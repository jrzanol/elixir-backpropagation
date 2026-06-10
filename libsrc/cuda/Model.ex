defmodule Model do
  def new(%{layers: layers, weights: weights, biases: biases}) do
    %{ref: CudaNif.new_model(layers, List.flatten(weights), List.flatten(biases))}
  end

  def load_train_batch(path), do: Dataset.load_flat_batch(path)
  def load_predict_batch(path), do: Dataset.load_flat_batch(path)

  def train_batch(model, batch, n_features, learn_rate) do
    features = Dataset.flatten_features(batch)
    labels = Dataset.flatten_labels(batch)

    :ok = CudaNif.train_batch(model.ref, features, labels, batch.count, n_features, learn_rate)
    model
  end

  def predict_batch(model, batch, n_features) do
    features = Dataset.flatten_features(batch)

    model.ref
    |> CudaNif.predict_batch(features, batch.count, n_features)
    |> Enum.map(&prob_to_class/1)
  end

  def train_epoch(model, batch_paths, n_features, learn_rate) do
    reduce_prefetched_batches(batch_paths, model, fn batch, model ->
      :ok =
        CudaNif.train_batch_binary(
          model.ref,
          batch.features_bin,
          batch.labels_bin,
          batch.count,
          n_features,
          learn_rate
        )

      model
    end)
  end

  def evaluate_paths(model, batch_paths, n_features) do
    reduce_prefetched_batches(batch_paths, Metrics.new(), fn batch, metrics ->
      predictions =
        model.ref
        |> CudaNif.predict_batch_binary(batch.features_bin, batch.count, n_features)
        |> Enum.map(&prob_to_class/1)

      Metrics.update(metrics, predictions, batch.labels)
    end)
  end

  defp reduce_prefetched_batches([], accumulator, _fun), do: accumulator

  defp reduce_prefetched_batches([first_path | remaining_paths], accumulator, fun) do
    first_batch = read_raw_batch!(first_path)

    {accumulator, pending_task} =
      Enum.reduce(remaining_paths, {accumulator, nil}, fn path, {accumulator, pending_task} ->
        next_task = Task.async(fn -> read_raw_batch!(path) end)
        batch = if pending_task, do: Task.await(pending_task, :infinity), else: first_batch
        {fun.(batch, accumulator), next_task}
      end)

    last_batch = if pending_task, do: Task.await(pending_task, :infinity), else: first_batch
    fun.(last_batch, accumulator)
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

  defp prob_to_class(prob), do: if(prob >= 0.5, do: 1, else: 0)
end
