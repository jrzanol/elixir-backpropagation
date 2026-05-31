defmodule Backprop.MixProject do
  use Mix.Project

  def project do
    [
      app: :backprop,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(),
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    case implementation() do
      "polyhok" ->
        [
          {:poly_hok, path: "deps/poly_hok"},
          {:nx, "~> 0.9.2"}
        ]

      _ ->
        []
    end
  end

  defp elixirc_paths do
    implementation = implementation()

    unless implementation in ["cuda", "polyhok"] do
      raise "BACKPROP_IMPL invalido: #{implementation}. Use cuda ou polyhok."
    end

    ["lib", "libsrc/#{implementation}"]
  end

  defp implementation do
    System.get_env("BACKPROP_IMPL", "polyhok")
  end
end
