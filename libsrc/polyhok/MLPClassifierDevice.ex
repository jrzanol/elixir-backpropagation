require PolyHok

PolyHok.defmodule_st MLPClassifierDevice do
  @moduledoc """
  Kernels e funcoes device da implementacao PolyHok.

  Este modulo nao prepara dados nem controla epochs. Ele apenas declara os
  kernels CUDA usados por `MLPClassifierHost`.
  """

  deft(sigmoid(float ~> float))

  defd sigmoid(x) do
    return(1.0 / (1.0 + exp(-x)))
  end

  deft(relu(float ~> float))

  defd relu(x) do
    zero = x - x
    result = zero

    if x > zero do
      result = x
    end

    return(result)
  end

  # Zerar gradientes e aplicar o update tratam pesos e biases num unico lancamento.
  # Cada PolyHok.spawn_st tem overhead de host (~ms); um lancamento por operacao (em
  # vez de um para pesos e outro para biases) reduz esse custo sem mudar a matematica.
  deft(zero_two_kernel(tfloat ~> integer ~> tfloat ~> integer ~> unit))

  defk zero_two_kernel(buf1, size1, buf2, size2) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < size1 do
      buf1[tid] = 0.0
    end

    if tid < size2 do
      buf2[tid] = 0.0
    end
  end

  deft(
    apply_update_two_kernel(
      tfloat ~> tfloat ~> integer ~> tfloat ~> tfloat ~> integer ~> float ~> integer ~> unit
    )
  )

  defk apply_update_two_kernel(p1, g1, size1, p2, g2, size2, lr, batch_count) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < size1 do
      p1[tid] = p1[tid] - lr / batch_count * g1[tid]
    end

    if tid < size2 do
      p2[tid] = p2[tid] - lr / batch_count * g2[tid]
    end
  end

  deft(
    train_batch_kernel(
      tfloat
      ~> tfloat
      ~> tfloat
      ~> tfloat
      ~> tfloat
      ~> tfloat
      ~> tinteger
      ~> tinteger
      ~> tinteger
      ~> tinteger
      ~> integer
      ~> integer
      ~> integer
      ~> unit
    )
  )

  defk train_batch_kernel(
         weights,
         biases,
         train_x,
         train_y,
         grad_w,
         grad_b,
         layers,
         weight_offsets,
         bias_offsets,
         neuron_offsets,
         layer_count,
         input_size,
         train_count
       ) do
    # Uma amostra por BLOCO; threads cooperam sobre os neuronios de cada camada.
    # act/delta em shared memory (uma copia por bloco), nao em arrays locais por thread,
    # que estouravam para memoria local (DRAM). Mesma otimizacao aplicada ao NIF CUDA.
    sample = blockIdx.x
    t = threadIdx.x
    nthreads = blockDim.x

    __shared__ act[512]
    __shared__ delta[512]

    if sample < train_count do
      for i in range(t, input_size, nthreads) do
        act[i] = train_x[sample * input_size + i]
      end
      __syncthreads()

      for l in range(1, layer_count, 1) do
        prev_size = layers[l - 1]
        curr_size = layers[l]
        w_off = weight_offsets[l]
        b_off = bias_offsets[l]
        prev_off = neuron_offsets[l - 1]
        curr_off = neuron_offsets[l]
        is_output = l == layer_count - 1

        for j in range(t, curr_size, nthreads) do
          net = biases[b_off + j]

          for i in range(0, prev_size, 1) do
            net = net + weights[w_off + i * curr_size + j] * act[prev_off + i]
          end

          if is_output do
            act[curr_off + j] = sigmoid(net)
          else
            act[curr_off + j] = relu(net)
          end
        end
        __syncthreads()
      end

      out_l = layer_count - 1
      out_size = layers[out_l]
      out_off = neuron_offsets[out_l]
      y = train_y[sample]

      for j in range(t, out_size, nthreads) do
        delta[out_off + j] = act[out_off + j] - y
      end
      __syncthreads()

      for rev in range(1, layer_count - 1, 1) do
        hl = layer_count - 1 - rev
        h_curr_size = layers[hl]
        h_next_size = layers[hl + 1]
        h_curr_off = neuron_offsets[hl]
        h_next_off = neuron_offsets[hl + 1]
        h_w_next_off = weight_offsets[hl + 1]

        for j in range(t, h_curr_size, nthreads) do
          sum = 0.0

          for k in range(0, h_next_size, 1) do
            sum = sum + weights[h_w_next_off + j * h_next_size + k] * delta[h_next_off + k]
          end

          if act[h_curr_off + j] > 0.0 do
            delta[h_curr_off + j] = sum
          else
            delta[h_curr_off + j] = 0.0
          end
        end
        __syncthreads()
      end

      for gl in range(1, layer_count, 1) do
        g_prev_size = layers[gl - 1]
        g_curr_size = layers[gl]
        g_prev_off = neuron_offsets[gl - 1]
        g_curr_off = neuron_offsets[gl]
        g_w_off = weight_offsets[gl]
        g_b_off = bias_offsets[gl]

        for j in range(t, g_curr_size, nthreads) do
          d = delta[g_curr_off + j]
          atomicAdd(grad_b + g_b_off + j, d)

          for i in range(0, g_prev_size, 1) do
            atomicAdd(grad_w + g_w_off + i * g_curr_size + j, d * act[g_prev_off + i])
          end
        end
      end
    end
  end

  deft(
    predict_batch_kernel(
      tfloat
      ~> tfloat
      ~> tfloat
      ~> tfloat
      ~> tinteger
      ~> tinteger
      ~> tinteger
      ~> tinteger
      ~> integer
      ~> integer
      ~> integer
      ~> unit
    )
  )

  defk predict_batch_kernel(
         weights,
         biases,
         batch_x,
         output,
         layers,
         weight_offsets,
         bias_offsets,
         neuron_offsets,
         layer_count,
         input_size,
         batch_count
       ) do
    sample = blockIdx.x
    t = threadIdx.x
    nthreads = blockDim.x

    __shared__ act[512]

    if sample < batch_count do
      for i in range(t, input_size, nthreads) do
        act[i] = batch_x[sample * input_size + i]
      end
      __syncthreads()

      for l in range(1, layer_count, 1) do
        prev_size = layers[l - 1]
        curr_size = layers[l]
        w_off = weight_offsets[l]
        b_off = bias_offsets[l]
        prev_off = neuron_offsets[l - 1]
        curr_off = neuron_offsets[l]
        is_output = l == layer_count - 1

        for j in range(t, curr_size, nthreads) do
          net = biases[b_off + j]

          for i in range(0, prev_size, 1) do
            net = net + weights[w_off + i * curr_size + j] * act[prev_off + i]
          end

          if is_output do
            act[curr_off + j] = sigmoid(net)
          else
            act[curr_off + j] = relu(net)
          end
        end
        __syncthreads()
      end

      out_l = layer_count - 1
      out_off = neuron_offsets[out_l]

      if t == 0 do
        output[sample] = act[out_off]
      end
    end
  end
end

PolyHok.include([MLPClassifierDevice])
