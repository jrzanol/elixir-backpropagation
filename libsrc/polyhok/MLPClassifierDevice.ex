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

  # Forward de uma camada oculta com ReLU: Cada thread calcula o net de UM neuronio j da camada atual.
  deft(forward_relu_kernel(tfloat ~> tfloat ~> tfloat ~> tfloat ~> integer ~> integer ~> unit))

  defk forward_relu_kernel(weights, input, biases, output, prev_size, curr_size) do
    j = blockIdx.x * blockDim.x + threadIdx.x

    if j < curr_size do
      net = biases[j]

      for i in range(0, prev_size, 1) do
        net = net + weights[i * curr_size + j] * input[i]
      end

      output[j] = relu(net)
    end
  end

  # Forward da camada de saida com sigmoide.
  deft(forward_sigmoid_kernel(tfloat ~> tfloat ~> tfloat ~> tfloat ~> integer ~> integer ~> unit))

  defk forward_sigmoid_kernel(weights, input, biases, output, prev_size, curr_size) do
    j = blockIdx.x * blockDim.x + threadIdx.x

    if j < curr_size do
      net = biases[j]

      for i in range(0, prev_size, 1) do
        net = net + weights[i * curr_size + j] * input[i]
      end

      output[j] = sigmoid(net)
    end
  end

  # delta_j^L = yhat_j - y_j   (cross-entropy + sigmoide colapsa para isso)
  deft(output_delta_kernel(tfloat ~> tfloat ~> tfloat ~> integer ~> unit))

  defk output_delta_kernel(yhat, y, delta, size) do
    j = blockIdx.x * blockDim.x + threadIdx.x

    if j < size do
      delta[j] = yhat[j] - y[j]
    end
  end

  deft(output_delta_scalar_kernel(tfloat ~> float ~> tfloat ~> integer ~> unit))

  defk output_delta_scalar_kernel(yhat, y, delta, size) do
    j = blockIdx.x * blockDim.x + threadIdx.x

    if j < size do
      delta[j] = yhat[j] - y
    end
  end

  # delta_j^l = ReLU'(a_j) * SUM_k( w_jk * delta_k^{l+1} )
  deft(hidden_delta_kernel(tfloat ~> tfloat ~> tfloat ~> tfloat ~> integer ~> integer ~> unit))

  defk hidden_delta_kernel(weights_next, delta_next, act_curr, delta_curr, curr_size, next_size) do
    j = blockIdx.x * blockDim.x + threadIdx.x

    if j < curr_size do
      sum = 0

      for k in range(0, next_size, 1) do
        sum = sum + weights_next[j * next_size + k] * delta_next[k]
      end

      a = act_curr[j]
      zero = a - a

      if a > zero do
        delta_curr[j] = sum
      else
        delta_curr[j] = 0
      end
    end
  end

  # dL/dw_ij = delta_j * a_i  (camada-l e camada-(l-1))
  deft(grad_weights_kernel(tfloat ~> tfloat ~> tfloat ~> integer ~> integer ~> unit))

  defk grad_weights_kernel(delta, act_prev, grad_w, prev_size, curr_size) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < prev_size * curr_size do
      i = tid / curr_size
      j = tid - i * curr_size
      grad_w[tid] = delta[j] * act_prev[i]
    end
  end

  # w <- w - eta * dL/dw   (SGD vanilla)
  deft(sgd_update_kernel(tfloat ~> tfloat ~> tfloat ~> tfloat ~> integer ~> unit))

  defk sgd_update_kernel(params, grads, updated, lr_arr, size) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < size do
      updated[tid] = params[tid] - lr_arr[0] * grads[tid]
    end
  end

  deft(sgd_update_scalar_kernel(tfloat ~> tfloat ~> tfloat ~> float ~> integer ~> unit))

  defk sgd_update_scalar_kernel(params, grads, updated, lr, size) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < size do
      updated[tid] = params[tid] - lr * grads[tid]
    end
  end

  deft(zero_kernel(tfloat ~> integer ~> unit))

  defk zero_kernel(buffer, size) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < size do
      buffer[tid] = 0.0
    end
  end

  deft(apply_mean_update_kernel(tfloat ~> tfloat ~> float ~> integer ~> integer ~> unit))

  defk apply_mean_update_kernel(params, grads, lr, batch_count, size) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < size do
      params[tid] = params[tid] - lr / batch_count * grads[tid]
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
         train_count,
         total_neurons
       ) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    act[512]
    delta[512]

    if tid < train_count do
      for i in range(0, input_size, 1) do
        act[i] = train_x[tid * input_size + i]
      end

      for l in range(1, layer_count, 1) do
        prev_size = layers[l - 1]
        curr_size = layers[l]
        w_off = weight_offsets[l]
        b_off = bias_offsets[l]
        prev_off = neuron_offsets[l - 1]
        curr_off = neuron_offsets[l]
        is_output = l == layer_count - 1

        for j in range(0, curr_size, 1) do
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
      end

      out_l = layer_count - 1
      out_size = layers[out_l]
      out_off = neuron_offsets[out_l]
      y = train_y[tid]

      for j in range(0, out_size, 1) do
        delta[out_off + j] = act[out_off + j] - y
      end

      for rev in range(1, layer_count - 1, 1) do
        hl = layer_count - 1 - rev
        h_curr_size = layers[hl]
        h_next_size = layers[hl + 1]
        h_curr_off = neuron_offsets[hl]
        h_next_off = neuron_offsets[hl + 1]
        h_w_next_off = weight_offsets[hl + 1]

        for j in range(0, h_curr_size, 1) do
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
      end

      for gl in range(1, layer_count, 1) do
        g_prev_size = layers[gl - 1]
        g_curr_size = layers[gl]
        g_prev_off = neuron_offsets[gl - 1]
        g_curr_off = neuron_offsets[gl]
        g_w_off = weight_offsets[gl]
        g_b_off = bias_offsets[gl]

        for j in range(0, g_curr_size, 1) do
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
         batch_count,
         total_neurons
       ) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    act[512]

    if tid < batch_count do
      for i in range(0, input_size, 1) do
        act[i] = batch_x[tid * input_size + i]
      end

      for l in range(1, layer_count, 1) do
        prev_size = layers[l - 1]
        curr_size = layers[l]
        w_off = weight_offsets[l]
        b_off = bias_offsets[l]
        prev_off = neuron_offsets[l - 1]
        curr_off = neuron_offsets[l]
        is_output = l == layer_count - 1

        for j in range(0, curr_size, 1) do
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
      end

      out_l = layer_count - 1
      out_off = neuron_offsets[out_l]
      output[tid] = act[out_off]
    end
  end
end

PolyHok.include([MLPClassifierDevice])
