defmodule Profiler do
  @moduledoc """
  Coleta tempos internos opcionais do fluxo de backpropagation.
  """

  @events [
    dataset_prepare: "Preparacao do dataset",
    dataset_materialize: "Materializacao dos batches",
    model_setup: "Inicializacao do modelo",
    load_train_batch: "Leitura de batch de treino",
    train_epoch: "Treino por epoca",
    train_batch: "Treino por batch",
    train_cpu_gpu_transfer: "Transferencia CPU/GPU no treino",
    train_gpu_compute: "Processamento GPU no treino",
    load_predict_batch: "Leitura de batch de predicao",
    predict_batch: "Predicao por batch",
    predict_cpu_gpu_transfer: "Transferencia CPU/GPU na predicao",
    predict_gpu_compute: "Processamento GPU na predicao",
    predict_gpu_cpu_transfer: "Transferencia GPU/CPU na predicao",
    metrics_update: "Atualizacao de metricas"
  ]

  @recorded_events_key {__MODULE__, :recorded_events}

  def new(enabled) do
    if enabled and is_nil(Process.get(@recorded_events_key)) do
      Process.put(@recorded_events_key, %{})
    end

    %{enabled?: enabled, events: %{}}
  end

  def enabled? do
    System.get_env("BACKPROP_PROFILE") in ["1", "true", "TRUE", "yes", "YES", "sim", "SIM"]
  end

  def measure(%{enabled?: true} = profile, event, fun) do
    {us, result} = :timer.tc(fun)
    {result, add(profile, event, us)}
  end

  def measure(profile, _event, fun), do: {fun.(), profile}

  def add(%{enabled?: true, events: events} = profile, event, us) do
    %{profile | events: Map.update(events, event, [us], &[us | &1])}
  end

  def add(profile, _event, _us), do: profile

  def runtime(event, fun) do
    if active?() do
      {us, result} = :timer.tc(fun)
      record(event, us)
      result
    else
      fun.()
    end
  end

  def record(event, us) do
    case Process.get(@recorded_events_key) do
      nil ->
        :ok

      events ->
        Process.put(@recorded_events_key, Map.update(events, event, [us], &[us | &1]))
        :ok
    end
  end

  defp active? do
    case Process.get(@recorded_events_key) do
      nil ->
        if enabled?() do
          Process.put(@recorded_events_key, %{})
          true
        else
          false
        end

      _events ->
        true
    end
  end

  def merge_recorded(%{enabled?: true} = profile) do
    recorded = Process.get(@recorded_events_key, %{})

    events =
      Enum.reduce(recorded, profile.events, fn {event, values}, acc ->
        Map.update(acc, event, values, &(values ++ &1))
      end)

    %{profile | events: events}
  end

  def merge_recorded(profile), do: profile

  def merge(%{enabled?: true} = left, %{events: right_events}) do
    merged =
      Enum.reduce(right_events, left.events, fn {event, values}, acc ->
        Map.update(acc, event, values, &(values ++ &1))
      end)

    %{left | events: merged}
  end

  def merge(left, _right), do: left

  def print(%{enabled?: true, events: events}) when map_size(events) > 0 do
    IO.puts("\n--- Tempos internos do backpropagation ---")

    Enum.each(@events, fn {event, label} ->
      values = events |> Map.get(event, []) |> Enum.reverse()

      if values != [] do
        print_event(label, values)
      end
    end)
  end

  def print(_profile), do: :ok

  def write_csv(%{enabled?: true, events: events}, path) do
    rows =
      events
      |> Enum.sort_by(fn {event, _values} -> Atom.to_string(event) end)
      |> Enum.flat_map(fn {event, values} ->
        values
        |> Enum.reverse()
        |> Enum.with_index(1)
        |> Enum.map(fn {us, occurrence} ->
          "#{event},#{occurrence},#{us}\n"
        end)
      end)

    File.mkdir_p!(Path.dirname(path))
    File.write!(path, ["event,occurrence,microseconds\n" | rows])
  end

  def write_csv(_profile, _path), do: :ok

  defp print_event(label, values) do
    count = length(values)
    total = Enum.sum(values)
    avg = total / count
    min_value = Enum.min(values)
    max_value = Enum.max(values)

    IO.puts(
      "#{label}: count=#{count} total=#{format_ms(total)} avg=#{format_ms(avg)} min=#{format_ms(min_value)} max=#{format_ms(max_value)}"
    )
  end

  defp format_ms(us), do: "#{Float.round(us / 1_000, 3)} ms"
end
