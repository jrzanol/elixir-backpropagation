defmodule Model do
  def new(initial_model) do
    initial_model
    |> MLPClassifierHost.to_gpu_model_state()
    |> validate_size()
  end

  def load_train_batch(path), do: Dataset.load_batch(path)
  def load_predict_batch(path), do: Dataset.load_batch(path)

  def train_batch(model, batch, _n_features, learn_rate) do
    MLPClassifierHost.train_batch_state(model, batch, learn_rate)
  end

  def predict_batch(model, batch, _n_features) do
    MLPClassifierHost.predict_batch_state(model, batch)
  end

  def train_epoch(model, batch_paths, _n_features, learn_rate) do
    reduce_prefetched_batches(batch_paths, model, fn batch, model ->
      MLPClassifierHost.train_binary_batch_state(model, batch, learn_rate)
    end)
  end

  def evaluate_paths(_model, [], _n_features), do: Metrics.new()

  def evaluate_paths(model, [first_path | remaining_paths], _n_features) do
    first_batch = read_raw_batch!(first_path)
    gpu_output = PolyHok.new_gnx(1, first_batch.count, {:f, 32})

    reduce_prefetched_batches(
      remaining_paths,
      Metrics.update(
        Metrics.new(),
        MLPClassifierHost.predict_binary_batch_state(model, first_batch, gpu_output),
        first_batch.labels
      ),
      fn batch, metrics ->
        predictions = MLPClassifierHost.predict_binary_batch_state(model, batch, gpu_output)
        Metrics.update(metrics, predictions, batch.labels)
      end
    )
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
      count: count,
      n_features: n_features
    }
  end

  defp validate_size(%{total_neurons: total_neurons} = model) when total_neurons <= 512, do: model

  defp validate_size(%{total_neurons: total_neurons}) do
    raise ArgumentError,
          "PolyHok train_batch_kernel suporta ate 512 neuronios totais; topologia atual tem #{total_neurons}"
  end
end
