defmodule Main do
  def run do
    seed = env_int("BACKPROP_SEED", 42)
    train_ratio = env_float("BACKPROP_TRAIN_RATIO", 0.8)
    epochs = env_int("BACKPROP_EPOCHS", 1000)
    learn_rate = env_float("BACKPROP_LEARN_RATE", 0.01)
    batch_size = env_int("BACKPROP_BATCH_SIZE", 1024)
    dataset = System.get_env("BACKPROP_DATASET") || Path.join("scripts", "prepared_dataset")

    IO.puts("Carregando dataset preparado...")

    streaming_dataset =
      Profiler.runtime(:dataset_prepare, fn ->
        Dataset.prepare(dataset,
          train_ratio: train_ratio,
          seed: seed
        )
      end)

    IO.puts(
      "Amostras: #{streaming_dataset.train_count + streaming_dataset.test_count}, Features: #{streaming_dataset.n_features}"
    )

    IO.puts("Treino: #{streaming_dataset.train_count}, Teste: #{streaming_dataset.test_count}")

    batch_store =
      Profiler.runtime(:dataset_materialize, fn ->
        Dataset.materialize_batches(streaming_dataset, batch_size)
      end)

    try do
      IO.puts("Batch size: #{batch_size}")

      IO.puts(
        "Batches treino: #{batch_store.train_batch_count}, teste: #{batch_store.test_batch_count}"
      )

      layers = Backpropagation.topology(streaming_dataset.n_features)
      IO.puts("Topologia: #{inspect(layers)}")

      IO.puts("\nTreinando #{epochs} epochs com lr=#{learn_rate}...")

      {train_us, %{train_metrics: train_metrics, test_metrics: test_metrics} = result} =
        :timer.tc(fn ->
          Backpropagation.run(%{
            dataset: dataset,
            seed: seed,
            train_ratio: train_ratio,
            epochs: epochs,
            learn_rate: learn_rate,
            batch_size: batch_size,
            layers: layers,
            n_features: streaming_dataset.n_features,
            train_batch_paths: batch_store.train_batch_paths,
            test_batch_paths: batch_store.test_batch_paths,
            train_count: streaming_dataset.train_count,
            test_count: streaming_dataset.test_count,
            profile: Profiler.enabled?()
          })
        end)

      trace("backpropagation_seconds=#{Float.round(train_us / 1_000_000, 3)}")

      print_results(train_metrics, test_metrics)
      profile = Map.get(result, :profile)
      Profiler.print(profile)

      case System.get_env("BACKPROP_PROFILE_FILE") do
        nil -> :ok
        path -> Profiler.write_csv(profile, path)
      end
    after
      Dataset.cleanup(batch_store)
    end
  end

  defp env_int(name, default) do
    case System.get_env(name) do
      nil -> default
      value -> String.to_integer(value)
    end
  end

  defp env_float(name, default) do
    case System.get_env(name) do
      nil -> default
      value -> String.to_float(value)
    end
  end

  defp print_results(train_metrics, test_metrics) do
    train = Metrics.summary(train_metrics)
    test = Metrics.summary(test_metrics)

    IO.puts("\n--- Resultados finais ---")
    IO.puts("Acuracia (treino) : #{Float.round(train.accuracy * 100.0, 2)}%")
    IO.puts("Acuracia (teste)  : #{Float.round(test.accuracy * 100.0, 2)}%")
    IO.puts("Precisao : #{Float.round(test.precision, 4)}")
    IO.puts("Recall   : #{Float.round(test.recall, 4)}")
    IO.puts("F1-Score : #{Float.round(test.f1, 4)}")
    IO.puts("Matriz:")
    IO.puts("  Real 0: TN=#{test.tn}  FP=#{test.fp}")
    IO.puts("  Real 1: FN=#{test.fn_val}  TP=#{test.tp}")
  end

  defp trace(message) do
    if System.get_env("BACKPROP_TRACE") == "1" do
      now = DateTime.utc_now() |> DateTime.to_iso8601()
      IO.puts("[TRACE] #{now} Main #{message}")
    end
  end
end
