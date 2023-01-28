# PicoGPT
You've seen [openai/gpt-2](https://github.com/openai/gpt-2).

You've seen [karpathy/minGPT](https://github.com/karpathy/mingpt).

You've even seen [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)!

But have you seen [picoGPT](https://github.com/jaymody/picoGPT)??!?

`picoGPT` is an unnecessarily tiny and minimal implementation of [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in plain [NumPy](https://numpy.org). The entire forward pass code is [40 lines of code](https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py#L3-L41).

So is picoGPT:
* Fast? ❌ Nah, picoGPT is megaSLOW 🐌
* Training code? ❌ Error, 4️⃣0️⃣4️⃣ not found
* Batch inference? ❌ picoGPT is civilized, one at a time only, single file line
* Support top-p? ❌ top-k? ❌ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Readable? 🤔 Well actually, it's not too horrible on the eyes!
* Smol??? ✅✅✅✅✅✅ YESS!!! TEENIE TINY in fact 🤏

#### Dependencies
```bash
pip install -r requirements.txt
```

If you're using an M1 Macbook, you'll need to replace `tensorflow` with `tensorflow-macos`.

Tested on `Python 3.9.10`.

#### Usage
```bash
python gpt2.py "Alan Turing theorized that computers would one day become"
```

Which generates

```
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

You can also control the number of tokens to generate, the model size (one of `["124M", "355M", "774M", "1558M"]`), and the directory to save the models:

```bash
python gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --model_size "124M" \
    --models_dir "models"
```

`gpt2_pico.py` is the same as `gpt2.py`, but has even fewer lines of code (removed comments, extra whitespace, and combined certain operations into a single line). Why? Because why not.
