defmodule Dataset do
  @moduledoc """
  Leitor de datasets pre-processados em batches binarios normalizados.
  """

  defmodule Batch do
    defstruct features: [], features_flat: nil, labels: [], count: 0, n_features: nil
  end

  defstruct path: nil,
            n_features: 0,
            train_count: 0,
            test_count: 0,
            train_ratio: 0.8,
            seed: 42,
            batch_size: 0,
            train_batch_count: 0,
            test_batch_count: 0,
            target_column: nil

  def prepare(path, opts \\ []) do
    metadata = read_metadata!(path)

    unless Map.get(metadata, "format") == "BPNORM1" do
      raise ArgumentError, "dataset preparado invalido: formato nao suportado"
    end

    train_ratio = Keyword.fetch!(opts, :train_ratio)
    seed = Keyword.fetch!(opts, :seed)
    stored_train_ratio = parse_float!(metadata, "train_ratio")
    stored_seed = parse_int!(metadata, "seed")

    if abs(stored_train_ratio - train_ratio) > 1.0e-12 do
      raise ArgumentError,
            "train_ratio=#{train_ratio} nao corresponde ao dataset preparado (#{stored_train_ratio})"
    end

    if stored_seed != seed do
      raise ArgumentError, "seed=#{seed} nao corresponde ao dataset preparado (#{stored_seed})"
    end

    %Dataset{
      path: path,
      n_features: parse_int!(metadata, "n_features"),
      train_count: parse_int!(metadata, "train_count"),
      test_count: parse_int!(metadata, "test_count"),
      train_ratio: stored_train_ratio,
      seed: stored_seed,
      batch_size: parse_int!(metadata, "batch_size"),
      train_batch_count: parse_int!(metadata, "train_batch_count"),
      test_batch_count: parse_int!(metadata, "test_batch_count"),
      target_column: metadata["target_column"]
    }
  end

  def materialize_batches(%Dataset{} = dataset, batch_size) do
    if batch_size != dataset.batch_size do
      raise ArgumentError,
            "batch_size=#{batch_size} nao corresponde ao dataset preparado (#{dataset.batch_size})"
    end

    %{
      train_batch_paths: batch_paths(dataset.path, "train"),
      test_batch_paths: batch_paths(dataset.path, "test"),
      train_batch_count: dataset.train_batch_count,
      test_batch_count: dataset.test_batch_count,
      train_count: dataset.train_count,
      test_count: dataset.test_count,
      temp_dir: nil
    }
  end

  def cleanup(_), do: :ok

  def load_batch(path) do
    case Path.extname(path) do
      ".bpbatch" -> load_binary_rows(path)
      _ -> path |> File.read!() |> :erlang.binary_to_term()
    end
  end

  def load_flat_batch(path) do
    case Path.extname(path) do
      ".bpbatch" -> load_binary_flat(path)
      _ -> load_batch(path)
    end
  end

  def flatten_features(%Batch{features_flat: flat}) when is_list(flat), do: flat
  def flatten_features(%Batch{features: features}), do: List.flatten(features)

  def flatten_labels(%Batch{labels: labels}), do: Enum.map(labels, &(&1 * 1.0))

  defp load_binary_rows(path) do
    {features_bin, labels, count, n_features} = read_binary_batch(path)

    %Batch{
      features: decode_feature_rows(features_bin, n_features),
      labels: labels,
      count: count,
      n_features: n_features
    }
  end

  defp load_binary_flat(path) do
    {features_bin, labels, count, n_features} = read_binary_batch(path)

    %Batch{
      features_flat: decode_floats(features_bin),
      labels: labels,
      count: count,
      n_features: n_features
    }
  end

  defp read_binary_batch(path) do
    <<"BPBATCH1", count::unsigned-little-32, n_features::unsigned-little-32, payload::binary>> =
      File.read!(path)

    feature_bytes = count * n_features * 4
    label_bytes = count * 4
    <<features_bin::binary-size(feature_bytes), labels_bin::binary-size(label_bytes)>> = payload

    {features_bin, decode_floats(labels_bin), count, n_features}
  end

  defp decode_feature_rows(features_bin, n_features) do
    row_bytes = n_features * 4

    for <<row_bin::binary-size(row_bytes) <- features_bin>> do
      decode_floats(row_bin)
    end
  end

  defp decode_floats(binary), do: for(<<value::float-little-32 <- binary>>, do: value)

  defp batch_paths(path, kind) do
    path
    |> Path.join(kind)
    |> Path.join("*.bpbatch")
    |> Path.wildcard()
    |> Enum.sort_by(&batch_index/1)
  end

  defp batch_index(path) do
    path
    |> Path.basename(".bpbatch")
    |> String.to_integer()
  end

  defp read_metadata!(path) do
    path
    |> Path.join("metadata.txt")
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "#")))
    |> Enum.reduce(%{}, fn line, acc ->
      case String.split(line, "=", parts: 2) do
        [key, value] -> Map.put(acc, key, value)
        _ -> acc
      end
    end)
  end

  defp parse_int!(metadata, key), do: metadata |> Map.fetch!(key) |> String.to_integer()
  defp parse_float!(metadata, key), do: metadata |> Map.fetch!(key) |> String.to_float()
end
