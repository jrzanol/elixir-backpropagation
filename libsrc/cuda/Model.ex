defmodule Model do
  def new(%{layers: layers, weights: weights, biases: biases}) do
    CudaNif.new_model(layers, List.flatten(weights), List.flatten(biases))
  end

  def load_train_batch(path), do: Dataset.load_flat_batch(path)
  def load_predict_batch(path), do: Dataset.load_flat_batch(path)

  def train_batch(model, batch, n_features, learn_rate) do
    features = Dataset.flatten_features(batch)
    labels = Dataset.flatten_labels(batch)

    :ok = CudaNif.train_batch(model, features, labels, batch.count, n_features, learn_rate)
    model
  end

  def predict_batch(model, batch, n_features) do
    features = Dataset.flatten_features(batch)

    model
    |> CudaNif.predict_batch(features, batch.count, n_features)
    |> Enum.map(&prob_to_class/1)
  end

  defp prob_to_class(prob), do: if(prob >= 0.5, do: 1, else: 0)
end
