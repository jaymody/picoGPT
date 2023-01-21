import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, h):
    x = linear(x, **c_attn)
    qkv = list(map(lambda x: np.split(x, h, axis=-1), np.split(x, 3, axis=-1)))
    casual_mask = (1 - np.tri(x.shape[0])) * -1e10
    heads = [attention(q, k, v, casual_mask) for q, k, v in zip(*qkv)]
    return linear(np.hstack(heads), **c_proj)

def block(x, mlp, attn, ln_1, ln_2, h):
    x = x + mha(layer_norm(x, **ln_1), **attn, h=h)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(ids, wte, wpe, blocks, ln_f, h):
    x = wte[ids] + wpe[np.arange(len(ids))]
    for block_params in blocks:
        x = block(x, **block_params, h=h)
    x = layer_norm(x, **ln_f)
    return x @ wte.T
