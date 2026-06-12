defmodule EquivalenceCase do
  def run do
    implementation = System.fetch_env!("BACKPROP_IMPL")
    output = System.fetch_env!("BACKPROP_TEST_OUTPUT")
    dataset_path = System.fetch_env!("BACKPROP_DATASET")
    train_ratio = env_float("BACKPROP_TRAIN_RATIO")
    epochs = env_int("BACKPROP_EPOCHS")
    learning_rate = env_float("BACKPROP_LEARN_RATE")
    batch_size = env_int("BACKPROP_BATCH_SIZE")
    seed = env_int("BACKPROP_SEED")

    File.mkdir_p!(output)
    dataset = Dataset.prepare(dataset_path, train_ratio: train_ratio, seed: seed)
    batches = Dataset.materialize_batches(dataset, batch_size)
    layers = Backpropagation.topology(dataset.n_features)
    model = layers |> Backpropagation.initial_model(seed) |> Model.new()

    {model, history} =
      Enum.reduce(1..epochs, {model, []}, fn epoch, {model, history} ->
        model =
          Model.train_epoch(model, batches.train_batch_paths, dataset.n_features, learning_rate)

        {_metrics, loss} = evaluate(model, batches.train_batch_paths, dataset.n_features, nil)
        IO.puts("[EPOCH_ERROR] impl=#{implementation} epoch=#{epoch} bce=#{format(loss)}")
        {model, [{epoch, loss} | history]}
      end)

    {train_metrics, _train_loss} =
      evaluate(model, batches.train_batch_paths, dataset.n_features, {output, "train"})

    {test_metrics, _test_loss} =
      evaluate(model, batches.test_batch_paths, dataset.n_features, {output, "test"})

    write_history(output, Enum.reverse(history))
    write_metrics(output, train_metrics, test_metrics)
    write_config(output, implementation, dataset_path, dataset, layers, epochs, learning_rate)
  end

  defp evaluate(model, paths, n_features, export) do
    handles = open_exports(export)

    try do
      {metrics, loss_sum, count} =
        Enum.reduce(paths, {Metrics.new(), 0.0, 0}, fn path, {metrics, loss_sum, count} ->
          batch = Model.load_predict_batch(path)
          probabilities = Model.predict_probabilities(model, batch, n_features)

          predictions =
            Enum.map(probabilities, fn probability -> if probability >= 0.5, do: 1, else: 0 end)

          write_predictions(handles, predictions, probabilities)

          batch_loss =
            Enum.zip(probabilities, batch.labels)
            |> Enum.reduce(0.0, fn {probability, label}, total ->
              probability = min(max(probability, 1.0e-7), 1.0 - 1.0e-7)

              total - label * :math.log(probability) -
                (1.0 - label) * :math.log(1.0 - probability)
            end)

          {
            Metrics.update(metrics, predictions, batch.labels),
            loss_sum + batch_loss,
            count + batch.count
          }
        end)

      {metrics, loss_sum / count}
    after
      close_exports(handles)
    end
  end

  defp open_exports(nil), do: nil

  defp open_exports({output, split}) do
    {
      File.open!(Path.join(output, "#{split}_predictions.bin"), [:write, :binary]),
      File.open!(Path.join(output, "#{split}_probabilities.f32"), [:write, :binary])
    }
  end

  defp write_predictions(nil, _predictions, _probabilities), do: :ok

  defp write_predictions({predictions_io, probabilities_io}, predictions, probabilities) do
    IO.binwrite(predictions_io, :erlang.list_to_binary(predictions))

    probability_binary =
      for probability <- probabilities, into: <<>>, do: <<probability::float-little-32>>

    IO.binwrite(probabilities_io, probability_binary)
  end

  defp close_exports(nil), do: :ok

  defp close_exports({predictions_io, probabilities_io}) do
    File.close(predictions_io)
    File.close(probabilities_io)
  end

  defp write_history(output, history) do
    rows = Enum.map(history, fn {epoch, loss} -> "#{epoch},#{format(loss)}\n" end)
    File.write!(Path.join(output, "epoch_error.csv"), ["epoch,bce\n" | rows])
  end

  defp write_metrics(output, train_metrics, test_metrics) do
    rows =
      for {split, metrics} <- [{"train", train_metrics}, {"test", test_metrics}] do
        summary = Metrics.summary(metrics)

        "#{split},#{format(summary.accuracy)},#{format(summary.precision)}," <>
          "#{format(summary.recall)},#{format(summary.f1)},#{summary.tn},#{summary.fp}," <>
          "#{summary.fn_val},#{summary.tp},#{summary.total}\n"
      end

    File.write!(
      Path.join(output, "metrics.csv"),
      ["split,accuracy,precision,recall,f1,tn,fp,fn,tp,total\n" | rows]
    )
  end

  defp write_config(output, implementation, dataset_path, dataset, layers, epochs, learning_rate) do
    File.write!(
      Path.join(output, "config.csv"),
      "implementation,dataset,train_ratio,seed,batch_size,epochs,learning_rate,topology\n" <>
        "#{implementation},#{dataset_path},#{format(dataset.train_ratio)},#{dataset.seed}," <>
        "#{dataset.batch_size},#{epochs},#{format(learning_rate)},\"#{inspect(layers)}\"\n"
    )
  end

  defp env_int(name), do: name |> System.fetch_env!() |> String.to_integer()
  defp env_float(name), do: name |> System.fetch_env!() |> String.to_float()
  defp format(value), do: :erlang.float_to_binary(value * 1.0, decimals: 12)
end

EquivalenceCase.run()
