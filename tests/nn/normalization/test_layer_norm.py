import torch
from ml_coding.nn.normalization.layer_norm import LayerNorm

def test_layer_norm():
    input_shape = (10,15,25)
    ln = LayerNorm(n_features=input_shape[-1])
    test_tensor = torch.rand(*input_shape)
    output = ln.forward(test_tensor)
    assert tuple(output.shape) == input_shape


# if __name__ == "__main__":
#     test_layer_norm()