# Shortformer

This repository contains the code and the final checkpoint of the Shortformer model. This file explains how to run our experiments on the WikiText-103 dataset. Read the full paper [here](https://arxiv.org/abs/2012.15832). 

The Shortformer is a combination of two methods:
1. **Staged Training**: We first train the model on short input subsequences and then train it on longer ones. This improves both train speed and evaluation perplexity.
2. **Position-Infused Attention + Caching**: We cache previously computed subsequence representations and attend to them using Position-Infused Attention. Position-Infused Attention modifies the model so that position embeddings are not added to the word embeddings at the bottom of the network, but instead, they are added to the keys and queries in the attention sublayer (but *not* to the values).
We show that PIA + caching vastly speeds up generation and also improves perplexity. 

Staged training requires no modification to the original code. To see how we implemented the Position-Infused Attention and caching, click [here](https://github.com/ofirpress/shortformer/commit/aa6786f84b788cbafd02e0914c57c99517a1a31c). 
Implementing PIA and caching is very easy, and we've provided detailed comments in the code to explain what how we did it. 

If you use this code or results from our paper, please cite:
```
@misc{press2020shortformer,
      title={Shortformer: Better Language Modeling using Shorter Inputs}, 
      author={Ofir Press and Noah A. Smith and Mike Lewis},
      year={2020},
      eprint={2012.15832},
}
```

## Requirements and Installation

This repository is a fork of the [Fairseq](https://github.com/pytorch/fairseq) repository and so has the same requirements. 

Once you've installed the dependencies, you can install this repository by running:

```bash
pip install --editable .
```

## Preparing the data

To download and preprocess the data, run:

```bash
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..


TEXT=examples/language_model/wikitext-103
python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

## Train/Inference for the different models

### Shortformer
Our Shortformer model takes the baseline and adds caching, Position-Infused Attention, and Staged Training.

To train the first stage:
```bash
python train.py --task language_modeling     data-bin/wikitext-103     --save-dir checkpoints128e100/     --arch transformer_lm_wiki103     --max-update 140100 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75     --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1     --criterion adaptive_loss --max-tokens 9216 --update-freq 1 --seed 1 --fp16     --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --tokens-per-sample 128 --max-tokens-valid 128 --tokens-from-prev 128 --curriculum 1000 --required-batch-size-multiple 1 --save-interval 100
```

If your GPUs don't have enough memory to execute that command, you can set --update-freq to 2 and --max-tokens to 4608, or set --update-freq to 3 and --max-tokens to 3072 for running the model with even lower memory constraints. This chunks the batch into 2 or 3 different parts and computes each part seperately (instead of in parallel), so it uses less memory but runs slower. 

After that, to train the model with the second (and final) stage:
```bash
python train.py --task language_modeling     data-bin/wikitext-103     --save-dir shortformer/ --restore-file checkpoints128e100/checkpoint100.pt     --arch transformer_lm_wiki103     --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75     --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1     --criterion adaptive_loss --max-tokens 9216 --update-freq 1 --seed 1 --fp16     --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --tokens-per-sample 512 --max-tokens-valid 512 --tokens-from-prev 512 --curriculum 1000 --required-batch-size-multiple 1 --no-epoch-checkpoints
```

Again, you can use the update-freq/max-tokens method from above if you run out of memory. 

#### Saved Checkpoint
If you'd like to download the Shortformer instead of training it, it is available [here](https://dl.fbaipublicfiles.com/shortformer/wikitext103-shortformer.pt). 
Rename that file to ```checkpoint_best.pt``` if you'd like to follow the directions below.

#### Inference

For nonoverlapping evaluation of the validation set, run:
```bash
fairseq-eval-lm data-bin/wikitext-103     --path shortformer/checkpoint_best.pt  --sample-break-mode none --gen-subset valid   --max-sentences 1
```

For token-by-token generation of the validation set, run:

```bash
fairseq-eval-lm data-bin/wikitext-103     --path shortformer/checkpoint_best.pt  --sample-break-mode none --gen-subset valid   --max-sentences 1 --sliding-inf 1 --context-window 511 --max-tokens 512
```

(Note that --context-window is a fairseq command and doesn't have the exact meaning that the term "context window" has in our paper.)

### Shortformer (without Staged Training)
Staged training improves the perplexity of the model *and* makes training faster, so there's no reason not to use it, but if you would like to train the Shortformer without it, the command is

```bash
python train.py --task language_modeling     data-bin/wikitext-103     --save-dir shortformer-no-st/      --arch transformer_lm_wiki103     --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75     --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1     --criterion adaptive_loss --max-tokens 9216 --update-freq 1 --seed 1 --fp16     --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --tokens-per-sample 512 --max-tokens-valid 512 --tokens-from-prev 512 --curriculum 1000 --required-batch-size-multiple 1 --no-epoch-checkpoints
```

For inference, use the same commands as the ones for the Shortformer (above).

### Baseline with Staged Training
Our Shortformer model is fast to train and for token-by-token generation, but if speed is not an issue, we can achieve slightly better performance by just applying Staged Training to the Baevski & Auli baseline LM. This model is very slow but achieves the best perplexity. 

To train the first stage, download the unmodified fairseq reporsitory and then run:
```bash
python train.py --task language_modeling     data-bin/wikitext-103     --save-dir checkpoints-st-128e50/     --arch transformer_lm_wiki103     --max-update 70050 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75     --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1     --criterion adaptive_loss --max-tokens 9216 --update-freq 1 --seed 1 --fp16     --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --tokens-per-sample 128  --required-batch-size-multiple 1 --save-interval 50
```

After that, to train the model with the second (and final) stage:
```bash
python train.py --task language_modeling     data-bin/wikitext-103     --save-dir st/ --restore-file checkpoints-st-128e50/checkpoint50.pt     --arch transformer_lm_wiki103     --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75     --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1     --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --seed 1 --fp16     --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --tokens-per-sample 3072  --no-epoch-checkpoints
```

#### Inference

For nonoverlapping evaluation of the validation set, run:

```bash
fairseq-eval-lm data-bin/wikitext-103     --path st/checkpoint_best.pt  --sample-break-mode none --gen-subset valid   --max-sentences 1
```

For sliding window evaluation of the validation set, with a stride of 2,560, run:

```bash
fairseq-eval-lm data-bin/wikitext-103     --path st/checkpoint_best.pt  --sample-break-mode none --gen-subset valid   --max-sentences 1 --context-window 2560
```

### Baseline - Baevski & Auli

To train the baseline, download the unmodified fairseq repository and then run:
```bash
python train.py --task language_modeling     data-bin/wikitext-103     --save-dir baseline/  --arch transformer_lm_wiki103     --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75     --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1     --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --seed 1 --fp16     --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --tokens-per-sample 3072  --no-epoch-checkpoints
```

#### Inference
Use the same commands as in the 'Baseline with Staged Training' inference subsection. 
