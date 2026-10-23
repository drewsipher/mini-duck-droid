"""Write a tiny, valid ONNX model without the `onnx` package (the sim venv does not have it).

Used for null-policy packages (recovery and open-loop episodic clips): the package format
requires a `policy.onnx`, and a null policy is "zero residual on top of the reference clip",
so the file is a real ONNX graph that maps an `obs` vector to a zero action vector. onnxruntime
loads it like any other policy. The encoder below is a hand-written protobuf writer for the few
ONNX message fields involved (ModelProto, GraphProto, NodeProto, ValueInfoProto, TensorProto).
"""
import struct


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _ld(field, payload):            # length-delimited (strings, bytes, sub-messages, packed)
    if isinstance(payload, str):
        payload = payload.encode()
    return _key(field, 2) + _varint(len(payload)) + payload


def _vi(field, value):              # varint field (int32/int64/enum)
    return _key(field, 0) + _varint(value)


def _dim(value):                    # TensorShapeProto.Dimension { int64 dim_value = 1 }
    return _ld(1, _vi(1, value))


def _value_info(name, shape, elem_type=1):
    shape_msg = b"".join(_dim(d) for d in shape)                  # TensorShapeProto { repeated Dimension dim = 1 }
    tensor_type = _vi(1, elem_type) + _ld(2, shape_msg)          # TypeProto.Tensor { elem_type = 1; shape = 2 }
    type_proto = _ld(1, tensor_type)                              # TypeProto { tensor_type = 1 }
    return _ld(1, name) + _ld(2, type_proto)                      # ValueInfoProto { name = 1; type = 2 }


def _float_tensor(name, dims, values):
    msg = b"".join(_vi(1, d) for d in dims)                       # TensorProto.dims = 1
    msg += _vi(2, 1)                                              # data_type = 2 (1 = FLOAT)
    msg += _ld(8, name)                                           # name = 8
    msg += _ld(9, struct.pack("<%df" % len(values), *values))     # raw_data = 9
    return msg


def write_zero_policy(path, obs_size, act_size, input_name="obs", output_name="continuous_actions"):
    """policy(obs[1, obs_size]) -> zeros[1, act_size]. Returns the path."""
    zeros = _ld(5, _float_tensor("zero_action", [1, act_size], [0.0] * act_size))      # GraphProto.initializer = 5
    node = _ld(1, "zero_action") + _ld(2, output_name) + _ld(3, "null_policy") + _ld(4, "Identity")   # NodeProto
    graph = _ld(1, node) + _ld(2, "null_policy") + zeros
    graph += _ld(11, _value_info(input_name, [1, obs_size]))      # GraphProto.input = 11
    graph += _ld(12, _value_info(output_name, [1, act_size]))     # GraphProto.output = 12
    opset = _ld(1, "") + _vi(2, 13)                               # OperatorSetIdProto { domain = 1; version = 2 }
    model = _vi(1, 7) + _ld(2, "graph_runtime.onnx_minimal") + _ld(7, graph) + _ld(8, opset)   # ModelProto
    with open(path, "wb") as f:
        f.write(model)
    return path


if __name__ == "__main__":
    import sys
    write_zero_policy(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
