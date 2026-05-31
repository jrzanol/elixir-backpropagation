defmodule Model do
  def new(initial_model) do
    initial_model
    |> MLPClassifierHost.to_gpu_model_state()
    |> validate_size()
  end

  def load_train_batch(path), do: Dataset.load_batch(path)
  def load_predict_batch(path), do: Dataset.load_batch(path)

  def train_batch(model, batch, _n_features, learn_rate) do
    MLPClassifierHost.train_batch_state(model, batch, learn_rate)
  end

  def predict_batch(model, batch, _n_features) do
    MLPClassifierHost.predict_batch_state(model, batch)
  end

  defp validate_size(%{total_neurons: total_neurons} = model) when total_neurons <= 512, do: model

  defp validate_size(%{total_neurons: total_neurons}) do
    raise ArgumentError,
          "PolyHok train_batch_kernel suporta ate 512 neuronios totais; topologia atual tem #{total_neurons}"
  end
end
