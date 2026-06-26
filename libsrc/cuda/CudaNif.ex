defmodule CudaNif do
  @moduledoc """
  Wrapper Elixir para o NIF Erlang que executa o MLP em CUDA. Elixir -> Erlang NIF -> C++/CUDA.
  """

  @on_load :load_nif

  def load_nif do
    nif_base =
      __DIR__
      |> Path.expand()
      |> Path.join("../../priv/CudaBackpropNif")
      |> Path.expand()

    :erlang.load_nif(String.to_charlist(nif_base), 0)
  end

  def new_model(_layers, _weights, _biases) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def train_batch(
        _model,
        _train_x_flat,
        _train_y,
        _batch_count,
        _input_size,
        _learn_rate
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def train_batch_binary(
        _model,
        _train_x_bin,
        _train_y_bin,
        _batch_count,
        _input_size,
        _learn_rate
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def predict_batch(
        _model,
        _x_flat,
        _batch_count,
        _input_size
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def predict_batch_binary(
        _model,
        _x_bin,
        _batch_count,
        _input_size
      ) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def last_timings(_model) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
