defmodule Profiler do
  @moduledoc """
  Coleta tempos internos opcionais do fluxo de backpropagation.
  """

  @events [
    model_setup: "Inicializacao do modelo",
    load_train_batch: "Leitura de batch de treino",
    train_epoch: "Treino por epoca",
    train_batch: "Treino por batch",
    load_predict_batch: "Leitura de batch de predicao",
    predict_batch: "Predicao por batch",
    metrics_update: "Atualizacao de metricas"
  ]

  def new(enabled), do: %{enabled?: enabled, events: %{}}

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
