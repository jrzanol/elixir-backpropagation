require PolyHok

defmodule MLPClassifierHost do
  @moduledoc """
  Codigo host da implementacao PolyHok.

  Este modulo prepara dados, aloca buffers GPU e dispara os kernels definidos em
  `MLPClassifierDevice`. A API usada pelo benchmark fica em `Model`.
  """

  def to_gpu_model_state(%{layers: layers, weights: weights, biases: biases}) do
    {weight_offsets, total_weights} =
      Enum.map_reduce(
        [
          0
          | Enum.map(Enum.zip(Enum.slice(layers, 0..-2//1), Enum.slice(layers, 1..-1//1)), fn {a,
                                                                                               b} ->
              a * b
            end)
        ],
        0,
        fn size, total -> {total, total + size} end
      )

    {bias_offsets, total_biases} =
      Enum.map_reduce([0 | tl(layers)], 0, fn size, total -> {total, total + size} end)

    {neuron_offsets, total_neurons} =
      Enum.map_reduce(layers, 0, fn size, total -> {total, total + size} end)

    flat_weights = List.flatten(weights)
    flat_biases = List.flatten(biases)

    %{
      layers: layers,
      total_weights: total_weights,
      total_biases: total_biases,
      total_neurons: total_neurons,
      gpu_weights:
        PolyHok.new_gnx(
          Nx.tensor([flat_weights], type: {:f, 32})
          |> Nx.reshape({1, total_weights})
        ),
      gpu_biases:
        PolyHok.new_gnx(Nx.tensor([flat_biases], type: {:f, 32}) |> Nx.reshape({1, total_biases})),
      gpu_grad_w: PolyHok.new_gnx(1, total_weights, {:f, 32}),
      gpu_grad_b: PolyHok.new_gnx(1, total_biases, {:f, 32}),
      gpu_layers:
        PolyHok.new_gnx(Nx.tensor([layers], type: {:s, 32}) |> Nx.reshape({1, length(layers)})),
      gpu_weight_offsets:
        PolyHok.new_gnx(
          Nx.tensor([weight_offsets], type: {:s, 32})
          |> Nx.reshape({1, length(weight_offsets)})
        ),
      gpu_bias_offsets:
        PolyHok.new_gnx(
          Nx.tensor([bias_offsets], type: {:s, 32})
          |> Nx.reshape({1, length(bias_offsets)})
        ),
      gpu_neuron_offsets:
        PolyHok.new_gnx(
          Nx.tensor([neuron_offsets], type: {:s, 32})
          |> Nx.reshape({1, length(neuron_offsets)})
        )
    }
  end

  def train_batch_state(state, %Dataset.Batch{} = batch, learning_rate) do
    flat_train_x = Enum.flat_map(batch.features, & &1)
    flat_train_y = Enum.map(batch.labels, fn y -> y * 1.0 end)

    gpu_train_x =
      PolyHok.new_gnx(
        Nx.tensor([flat_train_x], type: {:f, 32})
        |> Nx.reshape({1, length(flat_train_x)})
      )

    gpu_train_y =
      PolyHok.new_gnx(
        Nx.tensor([flat_train_y], type: {:f, 32})
        |> Nx.reshape({1, length(flat_train_y)})
      )

    # Zera gradientes dos pesos:
    PolyHok.spawn_st(
      &MLPClassifierDevice.zero_kernel/2,
      {div(state.total_weights + 255, 256), 1, 1},
      {256, 1, 1},
      [
        state.gpu_grad_w,
        state.total_weights
      ]
    )

    # Zera gradientes dos biases:
    PolyHok.spawn_st(
      &MLPClassifierDevice.zero_kernel/2,
      {div(state.total_biases + 255, 256), 1, 1},
      {256, 1, 1},
      [
        state.gpu_grad_b,
        state.total_biases
      ]
    )

    # Treina o batch:
    PolyHok.spawn_st(
      &MLPClassifierDevice.train_batch_kernel/14,
      {div(batch.count + 255, 256), 1, 1},
      {256, 1, 1},
      [
        state.gpu_weights,
        state.gpu_biases,
        gpu_train_x,
        gpu_train_y,
        state.gpu_grad_w,
        state.gpu_grad_b,
        state.gpu_layers,
        state.gpu_weight_offsets,
        state.gpu_bias_offsets,
        state.gpu_neuron_offsets,
        length(state.layers),
        hd(state.layers),
        batch.count,
        state.total_neurons
      ]
    )

    # Atualiza pesos:
    PolyHok.spawn_st(
      &MLPClassifierDevice.apply_mean_update_kernel/5,
      {div(state.total_weights + 255, 256), 1, 1},
      {256, 1, 1},
      [
        state.gpu_weights,
        state.gpu_grad_w,
        learning_rate,
        batch.count,
        state.total_weights
      ]
    )

    # Atualiza biases:
    PolyHok.spawn_st(
      &MLPClassifierDevice.apply_mean_update_kernel/5,
      {div(state.total_biases + 255, 256), 1, 1},
      {256, 1, 1},
      [state.gpu_biases, state.gpu_grad_b, learning_rate, batch.count, state.total_biases]
    )

    state
  end

  def predict_batch_state(state, %Dataset.Batch{} = batch) do
    flat_batch_x = Enum.flat_map(batch.features, & &1)

    gpu_batch_x =
      PolyHok.new_gnx(
        Nx.tensor([flat_batch_x], type: {:f, 32})
        |> Nx.reshape({1, length(flat_batch_x)})
      )

    gpu_output = PolyHok.new_gnx(1, batch.count, {:f, 32})

    # Prediz o batch:
    PolyHok.spawn_st(
      &MLPClassifierDevice.predict_batch_kernel/12,
      {div(batch.count + 255, 256), 1, 1},
      {256, 1, 1},
      [
        state.gpu_weights,
        state.gpu_biases,
        gpu_batch_x,
        gpu_output,
        state.gpu_layers,
        state.gpu_weight_offsets,
        state.gpu_bias_offsets,
        state.gpu_neuron_offsets,
        length(state.layers),
        hd(state.layers),
        batch.count,
        state.total_neurons
      ]
    )

    gpu_output
    |> PolyHok.get_gnx()
    |> Nx.to_flat_list()
    |> Enum.map(fn prob -> if prob >= 0.5, do: 1, else: 0 end)
  end
end
