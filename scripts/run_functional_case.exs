implementation = System.fetch_env!("BACKPROP_IMPL")
System.put_env("BACKPROP_DEBUG", "1")
System.put_env("BACKPROP_DEBUG_VALUES", "16")

initial_model = %{
  layers: [2, 2, 1],
  weights: [
    [0.5, -0.4, 0.3, 0.8],
    [0.7, -0.2]
  ],
  biases: [
    [0.1, -0.05],
    [0.05]
  ]
}

batch = %Dataset.Batch{
  features: [[0.6, 0.2]],
  labels: [1.0],
  count: 1,
  n_features: 2
}

model = Model.new(initial_model)
[probability_before] = Model.predict_probabilities(model, batch, 2)
model = Model.train_batch(model, batch, 2, 0.1)
[probability_after] = Model.predict_probabilities(model, batch, 2)

if Code.ensure_loaded?(MLPClassifierHost) do
  {weights, biases} = apply(MLPClassifierHost, :model_values, [model])
  format = fn values -> "[" <> Enum.map_join(values, ",", &:io_lib.format("~.9f", [&1])) <> "]" end
  IO.puts("[FUNCTIONAL_MODEL] impl=#{implementation} weights=#{format.(weights)} biases=#{format.(biases)}")
end

prediction_before = if probability_before >= 0.5, do: 1, else: 0
prediction_after = if probability_after >= 0.5, do: 1, else: 0

IO.puts(
  "[FUNCTIONAL] impl=#{implementation}" <>
    " probability_before=#{:erlang.float_to_binary(probability_before, decimals: 9)}" <>
    " prediction_before=#{prediction_before}" <>
    " probability_after=#{:erlang.float_to_binary(probability_after, decimals: 9)}" <>
    " prediction_after=#{prediction_after}"
)
