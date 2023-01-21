import argparse
import json
import os
import re

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder
from model import gpt2


def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    init_vars = tf.train.list_variables(tf_ckpt_path)
    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in init_vars:
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name.removeprefix("model/")
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def generate(ids, params, h, n_tokens_to_generate):
    max_seq_len = params["wpe"].shape[0]
    assert len(ids) + n_tokens_to_generate < max_seq_len

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(ids, **params, h=h)
        next_id = np.argmax(logits[-1])
        ids = np.append(ids, [next_id])

    return list(ids[len(ids) - n_tokens_to_generate :])


def main(prompt, models_dir, model_size, n_tokens_to_generate):
    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        download_gpt2_files(model_size, model_dir)

    with open(os.path.join(model_dir, "hparams.json")) as file:
        hparams = json.load(file)

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)
    encoder = get_encoder(model_size, models_dir)
    input_ids = [encoder.encoder["<|endoftext|>"]] if prompt is None else encoder.encode(prompt)
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate text with GPT-2.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input text to condition the outputs. If not set, we'll generate unconditioned (i.e. start with <|endoftext|> token).",
        default=None,
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Base directory for the model directories.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="124M",
        help="Model size. Must be one of ['124M', '355M', '774M', '1558M']",
    )
    parser.add_argument(
        "--n_tokens_to_generate",
        type=int,
        default=40,
        help="Number of tokens to generate.",
    )
    args = parser.parse_args()

    print(main(**args.__dict__))
