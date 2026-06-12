defmodule Model do
  def new(initial_model) do
    initial_model
    |> MLPClassifierHost.to_gpu_model_state()
    |> validate_size()
  end

  def load_train_batch(path), do: Dataset.load_batch(path)
  def load_predict_batch(path), do: Dataset.load_batch(path)

  def train_batch(model, batch, _n_features, learn_rate) do
    Profiler.runtime(:train_batch, fn ->
      MLPClassifierHost.train_batch_state(model, batch, learn_rate)
    end)
  end

  def predict_batch(model, batch, _n_features) do
    Profiler.runtime(:predict_batch, fn ->
      MLPClassifierHost.predict_batch_state(model, batch)
    end)
  end

  def predict_probabilities(model, batch, _n_features) do
    Profiler.runtime(:predict_batch, fn ->
      MLPClassifierHost.predict_probabilities_batch_state(model, batch)
    end)
  end

  def train_epoch(model, batch_paths, _n_features, learn_rate) do
    reduce_prefetched_batches(batch_paths, model, :load_train_batch, fn batch, model ->
      Profiler.runtime(:train_batch, fn ->
        MLPClassifierHost.train_binary_batch_state(model, batch, learn_rate)
      end)
    end)
  end

  def evaluate_paths(_model, [], _n_features), do: Metrics.new()

  def evaluate_paths(model, [first_path | remaining_paths], _n_features) do
    first_timed_batch = timed_read_raw_batch!(first_path)
    first_batch = record_load(first_timed_batch, :load_predict_batch)
    gpu_output = PolyHok.new_gnx(1, first_batch.count, {:f, 32})

    probabilities = predict_binary_probabilities(model, first_batch, gpu_output)

    reduce_prefetched_batches(
      remaining_paths,
      Metrics.update(
        Metrics.new(),
        Enum.map(probabilities, &prob_to_class/1),
        first_batch.labels
      ),
      :load_predict_batch,
      fn batch, metrics ->
        probabilities = predict_binary_probabilities(model, batch, gpu_output)
        Metrics.update(metrics, Enum.map(probabilities, &prob_to_class/1), batch.labels)
      end
    )
  end

  def predict_binary_probabilities(model, batch, gpu_output \\ nil) do
    output = gpu_output || PolyHok.new_gnx(1, batch.count, {:f, 32})

    Profiler.runtime(:predict_batch, fn ->
      MLPClassifierHost.predict_binary_probabilities_batch_state(model, batch, output)
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
      count: count,
      n_features: n_features
    }
  end

  defp validate_size(%{total_neurons: total_neurons} = model) when total_neurons <= 512, do: model

  defp validate_size(%{total_neurons: total_neurons}) do
    raise ArgumentError,
          "PolyHok train_batch_kernel suporta ate 512 neuronios totais; topologia atual tem #{total_neurons}"
  end

  defp prob_to_class(prob), do: if(prob >= 0.5, do: 1, else: 0)
end
