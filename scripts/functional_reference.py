#!/usr/bin/env python3
"""Referencia float32 para o caso funcional pequeno."""

from __future__ import annotations

import math
import struct


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sigmoid(value: float) -> float:
    return f32(1.0 / (1.0 + math.exp(-value)))


def forward(weights: list[float], biases: list[float]) -> tuple[list[float], float]:
    features = [f32(0.6), f32(0.2)]
    hidden: list[float] = []
    for neuron in range(2):
        net = biases[neuron]
        for feature in range(2):
            net = f32(net + f32(weights[feature * 2 + neuron] * features[feature]))
        hidden.append(f32(max(0.0, net)))

    net = biases[2]
    for neuron in range(2):
        net = f32(net + f32(weights[4 + neuron] * hidden[neuron]))
    return hidden, sigmoid(net)


def values(items: list[float]) -> str:
    return "[" + ",".join(f"{item:.9f}" for item in items) + "]"


def main() -> None:
    weights = [f32(value) for value in [0.5, -0.4, 0.3, 0.8, 0.7, -0.2]]
    biases = [f32(value) for value in [0.1, -0.05, 0.05]]
    features = [f32(0.6), f32(0.2)]

    print(
        "[DEBUG_SNAPSHOT] impl=python epoch=0"
        f" weights={values(weights)} biases={values(biases)} grad_w=[] grad_b=[]"
    )

    hidden, probability_before = forward(weights, biases)
    output_delta = f32(probability_before - 1.0)
    hidden_delta = [
        f32(f32(weights[4 + index] * output_delta) if hidden[index] > 0.0 else 0.0)
        for index in range(2)
    ]
    deltas = hidden_delta + [output_delta]
    grad_w = [
        f32(hidden_delta[0] * features[0]),
        f32(hidden_delta[1] * features[0]),
        f32(hidden_delta[0] * features[1]),
        f32(hidden_delta[1] * features[1]),
        f32(output_delta * hidden[0]),
        f32(output_delta * hidden[1]),
    ]

    learning_rate = f32(0.1)
    updated_weights = [f32(weight - f32(learning_rate * grad)) for weight, grad in zip(weights, grad_w)]
    updated_biases = [f32(bias - f32(learning_rate * delta)) for bias, delta in zip(biases, deltas)]
    _, probability_after = forward(updated_weights, updated_biases)

    print(
        "[DEBUG_SNAPSHOT] impl=python epoch=1"
        f" weights={values(updated_weights)} biases={values(updated_biases)}"
        f" grad_w={values(grad_w)} grad_b={values(deltas)}"
    )
    print(
        "[FUNCTIONAL] impl=python"
        f" probability_before={probability_before:.9f} prediction_before={int(probability_before >= 0.5)}"
        f" probability_after={probability_after:.9f} prediction_after={int(probability_after >= 0.5)}"
    )


if __name__ == "__main__":
    main()
