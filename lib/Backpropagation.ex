defmodule Backpropagation do
  def topology(n_features) when n_features > 0 and n_features + 385 <= 512 do
    [n_features, 256, 128, 1]
  end

  def topology(n_features) do
    raise ArgumentError,
          "topologia [#{n_features}, 256, 128, 1] excede o limite de 512 neuronios do PolyHok"
  end

  def run(
        %{
          seed: seed,
          epochs: epochs,
          learn_rate: learn_rate,
          layers: layers,
          n_features: n_features,
          train_batch_paths: train_batch_paths,
          test_batch_paths: test_batch_paths
        } = config
      ) do
    profile = Profiler.new(Map.get(config, :profile, false))

    {model, profile} =
      Profiler.measure(profile, :model_setup, fn ->
        layers
        |> initial_model(seed)
        |> Model.new()
      end)

    {train_us, {model, profile}} =
      :timer.tc(fn ->
        train(model, train_batch_paths, epochs, learn_rate, n_features, profile)
      end)

    {train_metrics, profile} = evaluate(model, train_batch_paths, n_features, profile)
    {test_metrics, profile} = evaluate(model, test_batch_paths, n_features, profile)
    profile = Profiler.merge_recorded(profile)

    %{
      train_metrics: train_metrics,
      test_metrics: test_metrics,
      train_microseconds: train_us,
      profile: profile
    }
  end

  defp train(model, train_batch_paths, epochs, learn_rate, n_features, profile) do
    Enum.reduce(1..epochs, {model, profile}, fn epoch, {model, profile} ->
      epoch_start = System.monotonic_time()

      {model, profile} = train_epoch(model, train_batch_paths, n_features, learn_rate, profile)

      profile =
        Profiler.add(
          profile,
          :train_epoch,
          System.convert_time_unit(System.monotonic_time() - epoch_start, :native, :microsecond)
        )

      if rem(epoch, 100) == 0 do
        IO.puts("Epoch #{epoch}/#{epochs} concluida")
      end

      {model, profile}
    end)
  end

  defp train_epoch(model, train_batch_paths, n_features, learn_rate, profile) do
    if function_exported?(Model, :train_epoch, 4) do
      {Model.train_epoch(model, train_batch_paths, n_features, learn_rate), profile}
    else
      Enum.reduce(train_batch_paths, {model, profile}, fn path, {model, profile} ->
        {batch, profile} =
          Profiler.measure(profile, :load_train_batch, fn ->
            Model.load_train_batch(path)
          end)

        {model, profile} =
          Profiler.measure(profile, :train_batch, fn ->
            Model.train_batch(model, batch, n_features, learn_rate)
          end)

        {model, profile}
      end)
    end
  end

  defp evaluate(model, batch_paths, n_features, profile) do
    if function_exported?(Model, :evaluate_paths, 3) do
      {Model.evaluate_paths(model, batch_paths, n_features), profile}
    else
      Enum.reduce(batch_paths, {Metrics.new(), profile}, fn path, {metrics, profile} ->
        {batch, profile} =
          Profiler.measure(profile, :load_predict_batch, fn ->
            Model.load_predict_batch(path)
          end)

        {predictions, profile} =
          Profiler.measure(profile, :predict_batch, fn ->
            Model.predict_batch(model, batch, n_features)
          end)

        {metrics, profile} =
          Profiler.measure(profile, :metrics_update, fn ->
            Metrics.update(metrics, predictions, batch.labels)
          end)

        {metrics, profile}
      end)
    end
  end

  def initial_model(layers, seed) do
    layer_pairs =
      Enum.zip(
        Enum.slice(layers, 0..-2//1),
        Enum.slice(layers, 1..-1//1)
      )

    {layer_params, _rng} =
      Enum.map_reduce(layer_pairs, XorShift64Star.seed(seed), fn {prev_size, curr_size}, rng ->
        limit = :math.sqrt(6.0 / prev_size)

        {weights, next_rng} =
          Enum.map_reduce(1..(prev_size * curr_size), rng, fn _, current_rng ->
            {new_rng, u} = XorShift64Star.nextf(current_rng)
            {(u * 2.0 - 1.0) * limit, new_rng}
          end)

        biases = List.duplicate(0.0, curr_size)
        {{weights, biases}, next_rng}
      end)

    {weights, biases} = Enum.unzip(layer_params)
    %{layers: layers, weights: weights, biases: biases}
  end
end
