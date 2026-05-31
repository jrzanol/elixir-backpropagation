import Bitwise

defmodule XorShift64Star do
  @moduledoc """
  PRNG xorshift64* usado pelo projeto base para inicializar pesos.
  """

  @mask 0xFFFFFFFFFFFFFFFF
  @mul 2_685_821_657_736_338_717

  def seed(s) when s == 0, do: 1
  def seed(s), do: s &&& @mask

  def next(state) do
    x = state
    x = bxor(x, x >>> 12)
    x = bxor(x, x <<< 25 &&& @mask)
    x = bxor(x, x >>> 27)
    x = x &&& @mask
    output = x * @mul &&& @mask
    {x, output}
  end

  def nextf(state) do
    {new_state, v} = next(state)
    top53 = v >>> 11
    float_val = top53 * (1.0 / :math.pow(2, 53))
    {new_state, float_val}
  end
end
