defmodule Metrics do
  @moduledoc """
  Acumulador streaming para metricas de classificacao binaria.
  """

  defstruct total: 0, correct: 0, tp: 0, tn: 0, fp: 0, fn_val: 0

  def new, do: %Metrics{}

  def update(%Metrics{} = metrics, predictions, labels) do
    Enum.zip(predictions, labels)
    |> Enum.reduce(metrics, fn {pred, truth}, acc ->
      update_one(acc, pred, truth)
    end)
  end

  def summary(%Metrics{} = metrics) do
    precision =
      if metrics.tp + metrics.fp > 0 do
        metrics.tp / (metrics.tp + metrics.fp)
      else
        0.0
      end

    recall =
      if metrics.tp + metrics.fn_val > 0 do
        metrics.tp / (metrics.tp + metrics.fn_val)
      else
        0.0
      end

    f1 =
      if precision + recall > 0 do
        2.0 * precision * recall / (precision + recall)
      else
        0.0
      end

    accuracy =
      if metrics.total > 0 do
        metrics.correct / metrics.total
      else
        0.0
      end

    Map.merge(Map.from_struct(metrics), %{
      accuracy: accuracy,
      precision: precision,
      recall: recall,
      f1: f1
    })
  end

  defp update_one(%Metrics{} = acc, pred, truth) do
    t = trunc(truth)
    correct = if pred == t, do: 1, else: 0

    acc = %{acc | total: acc.total + 1, correct: acc.correct + correct}

    cond do
      t == 1 and pred == 1 -> %{acc | tp: acc.tp + 1}
      t == 0 and pred == 0 -> %{acc | tn: acc.tn + 1}
      t == 0 and pred == 1 -> %{acc | fp: acc.fp + 1}
      t == 1 and pred == 0 -> %{acc | fn_val: acc.fn_val + 1}
    end
  end
end
