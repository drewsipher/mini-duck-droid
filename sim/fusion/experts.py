"""Tracking experts as plain numpy MLPs, read out of their policy.onnx.

The tracking ONNX is: (obs - mean) * (1/std) -> 3 swish layers -> linear -> first half -> tanh
(playground/common/export_onnx.py). Holding the weights as arrays lets the fusion tools run
several experts per step cheaply, blend their outputs, or blend their weights.
"""
import numpy as np
import onnx
from onnx import numpy_helper


class MLP:
    def __init__(self, mean, inv_std, layers):
        self.mean, self.inv_std, self.layers = mean, inv_std, layers     # layers: [(W, b)], W is (in, out)

    @classmethod
    def from_onnx(cls, path):
        init = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in onnx.load(path).graph.initializer}
        mean = next(v for k, v in init.items() if k.endswith("sub/ReadVariableOp:0"))
        inv_std = next(v for k, v in init.items() if "truediv_recip" in k)
        layers, i = [], 0
        while any(f"hidden_{i}_" in k for k in init):
            W = next(v for k, v in init.items() if f"hidden_{i}_" in k and "Cast" in k)
            b = next(v for k, v in init.items() if f"hidden_{i}_" in k and "BiasAdd" in k)
            layers.append((W, b)); i += 1
        return cls(mean, inv_std, layers)

    def __call__(self, obs):
        x = (np.asarray(obs, np.float64) - self.mean) * self.inv_std
        for j, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if j < len(self.layers) - 1:
                x = x * 0.5 * (1.0 + np.tanh(0.5 * x))                     # swish, x * sigmoid(x)
        return np.tanh(x[..., : x.shape[-1] // 2])

    def blend(self, other, w):
        """Weight-space interpolation (1 - w) * self + w * other, normaliser included."""
        mix = lambda a, b: (1 - w) * a + w * b
        return MLP(mix(self.mean, other.mean), mix(self.inv_std, other.inv_std),
                   [(mix(W0, W1), mix(b0, b1)) for (W0, b0), (W1, b1) in zip(self.layers, other.layers)])


if __name__ == "__main__":      # check against onnxruntime
    import sys
    import onnxruntime as ort
    p = sys.argv[1]
    m, s = MLP.from_onnx(p), ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(64, len(m.mean)))
    ref = np.concatenate([s.run(None, {s.get_inputs()[0].name: r[None].astype(np.float32)})[0] for r in x])
    print("max abs diff vs onnxruntime:", float(np.abs(m(x) - ref).max()))
