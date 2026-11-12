<p align="center">
  <img src="assets/logo.png" alt="Gu-1B project mascot" width="260">
</p>

<h1 align="center">Gu-1B-500M-tokens</h1>
<p align="center"><strong>Anewo's first experimental language model</strong></p>

Gu-1B is the first experimental model from **Anewo**, our startup. We built and trained it from scratch to understand language-model training through direct experimentation, share the process and give other learners a checkpoint they can inspect and build on.

This release contains a **1,009,866,624-parameter model trained on 500 million tokens**. It is an early experiment with documented weaknesses. Our aim is to make this checkpoint a learning resource and a starting point for further work with larger models as training compute becomes available.

[Model repository and weights on Hugging Face](https://huggingface.co/tony6767/Gu-1B-500M-tokens)

## Why we paused at 500M tokens

We trained on a single **Colab A100 with 80 GB GPU memory**, using a limited balance of paid compute units. The experiment initially considered 2B training tokens, was reduced to a 1B-token goal, and reached its first full resumable checkpoint at 500M.

Observed throughput was approximately **11.4K-11.5K tokens per second** at 4K sequence length. When we assessed the remaining budget, the session showed **79.43 compute units** and an approximate usage rate of **8.88 units per hour**. That covered about **8 hours 57 minutes**. Resuming the saved 500M checkpoint and training another 500M targets at 11.4K tokens per second would require about **12 hours 11 minutes**, before checkpoint writes and the planned longer-context stages.

We chose to preserve the 500M checkpoint, evaluate it and share it. **The stopping decision reflects our compute budget.** Further pretraining remains part of the experiment. These costs and speeds describe our observed Colab session and can change across runtimes.

Training briefly continued beyond 500M before we manually interrupted it. This release uses the completed 500M save. Resuming it repeats any work performed after that save.

## What this experiment is for

- Learn how a decoder-only transformer, tokenizer and training loop work together.
- Study how generation changes across checkpoints and token budgets.
- Practice recovering training after a Colab session ends.
- Compare held-out prediction metrics with factual accuracy and coherence in generated text.
- Experiment with further pretraining, data choices and instruction tuning.
- Build practical experience for Anewo's future experiments with larger models.

Training a larger model will require a compatible architecture and a new training plan. This checkpoint offers a reusable implementation and experimental reference. Its tensors belong to the Gu-1B architecture, and changing the parameter count requires explicit weight-transfer or initialization work.

## What has been completed

| Item | Status |
|---|---|
| From-scratch Gu-1B training | Reached the 500M-token saved checkpoint |
| Intermediate model snapshots | Save confirmations observed at 100M, 200M, 300M and 400M |
| Full 500M training-state checkpoint | Saved to Google Drive |
| Model export | Approximately 2.02 GB BF16 weights and metadata |
| Held-out evaluation | One 32,768-target evaluation completed |
| Generation tests | Readable text with factual and context-consistency failures |
| Continued pretraining toward 1B tokens | Future experiment, subject to compute |
| Instruction tuning and larger-model experiments | Proposed work |

The full checkpoint's public upload should be checked in the Hugging Face file list. Saving it to Drive and publishing it on the Hub are separate steps.

## Model and training


| Setting | Value |
|---|---|
| Parameters | 1,009,866,624 |
| Layers | 27 |
| Hidden width | 1,536 |
| Query / key-value heads | 24 / 8 |
| Feed-forward width | 6,144 |
| Vocabulary size | 49,152 |
| Embeddings | Shared input and output weights |
| Training data | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` configuration |
| Tokenizer | From `HuggingFaceTB/SmolLM-135M` |
| Training targets consumed | 500,000,000 |
| Sequence length used through this checkpoint | 4,096 tokens |
| Hardware used | Single Colab A100 with 80 GB GPU memory |
| Optimizer | AdamW |
| Training precision | BF16 autocast with FP32 parameters |
| Standard update size | 65,536 target tokens, shortened at save boundaries |
| Export precision | BF16 |

The tokenizer and corpus commit IDs are recorded in checkpoint metadata. The model was initialized randomly; the SmolLM repository supplies its tokenizer. The architecture is configured for up to 32,768 positions. Longer-context training stages had not been reached at 500M tokens, so this release's long-context ability remains unmeasured.

## Preliminary evaluation

We evaluated the saved 500M checkpoint on eight 4,096-token sequences from held-out documents in the training corpus. Every hundredth document is reserved before tokenization.

| Metric | Result |
|---|---|
| Evaluated next-token targets | 32,768 |
| Mean cross-entropy loss | 2.237 |
| Perplexity | 9.37 |

These are results from a single small evaluation. Comparisons need the same tokenizer, data selection and evaluation procedure. Broader benchmarks and repeated-run studies remain future work.

### Observed limitations

Sampled completions showed grammatical sentences alongside factual mistakes, repetition, contradictions and topic drift. The model failed prompts about the two possible bit values, why plants need sunlight, and identifying the owner of a red bicycle from facts provided in the prompt.

Gu-1B is a base next-token prediction model. Its instruction-following behavior, factual accuracy, reasoning, coding and practical long-context use need further training and evaluation. The reported checkpoint only reached **4,096-token training sequences**. The planned 16K and 32K training stages had not been reached. Generated information needs independent verification.

## Files to upload to GitHub

| Path | Purpose |
|---|---|
| `README.md` | Experiment, compute constraints, results and instructions |
| `assets/logo.png` | Supplied project logo displayed at the top of this README |
| `training_Gu_1B.ipynb` | Architecture, training, checkpoint recovery, evaluation and generation |
| `verify_gu_notebook.py` | Small-model regression checks |
| `.gitignore` | Excludes weights, caches and local credentials |

The notebook code lives here. Large model files belong in the linked Hugging Face model repository.

## Generate with the 500M model

1. Open `training_Gu_1B.ipynb` in Google Colab.
2. Run setup, model configuration, model architecture and checkpoint-utility definition cells. Complete Drive authorization if setup requests it.
3. In the final generation cell, replace its `MODEL_FILE` assignment with the code below.
4. Run that generation cell to load the model and generate text. The training cell starts a training workload, so use the definition and generation cells when evaluating this release.

```python
from huggingface_hub import hf_hub_download

MODEL_FILE = hf_hub_download(
    repo_id="tony6767/Gu-1B-500M-tokens",
    filename="Gu-1B-500000000-model.pt",
)
```

Once the generation cell has loaded the model, reuse it with an interactive prompt:

```python
prompt = input("Enter your prompt: ")
print(generate_gu(inference_model, tokenizer, prompt, max_new_tokens=100))
```

The model uses the notebook's custom `GuModel` implementation. A Transformers `AutoModelForCausalLM.from_pretrained()` interface requires additional integration. CUDA inference chooses native BF16 where supported and FP16 otherwise. Attention allows a math fallback for GPUs such as T4. Start with short prompts because the fallback and prefix recomputation increase memory use and latency. CPU inference uses FP32.

## Learn from the checkpoints

| Artifact | Contents and use |
|---|---|
| `Gu-1B-500000000-model.pt` | BF16 model export for generation and inspection |
| Full `checkpoint.pt` plus `manifest.json` | Model, optimizer, RNG and data state for continuing training |
| Intermediate snapshots | Weights for studying changes during training |

The 100M-400M snapshots omit optimizer and streaming state. Resuming the complete training process needs the full checkpoint. Check the Hub file list for `training-checkpoint-500M/checkpoint.pt` and `training-checkpoint-500M/manifest.json` before downloading full training state. Preserve the downloaded manifest and checkpoint together.

Source revisions and library versions are recorded in checkpoint metadata. Full files contain serialized Python and NumPy objects, so load full training-state files from trusted sources. Model exports load with `weights_only=True`.

## Reproduce or continue the experiment

For a fresh training experiment, choose an A100 runtime, use a dedicated checkpoint folder and set `RESUME_FROM=None` before invoking training. To reproduce the release milestone, set:

```python
TARGET_TOKENS = 1_000_000_000
SESSION_STOP_TOKENS = 500_000_000
RESUME_FROM = None
```

For continuation, set `RESUME_FROM` to the full checkpoint file. Clear the previous stop before starting the training cell:

```python
TARGET_TOKENS = 1_000_000_000
SESSION_STOP_TOKENS = None
RESUME_FROM = "/content/path-to-full-checkpoint/checkpoint.pt"
```

Replace that example path with the downloaded full checkpoint. `RESUME_FROM="auto"` is also available for a checkpoint root containing a single experiment. Keep architecture and optimizer settings compatible with the save. Review the source revisions, library versions and session limits.

The current default schedule uses 4K sequences through 900M targets, 16K through 980M, then 32K. Model snapshots occur every 100M targets. Full state is saved every 500M targets and at completed planned session stops. Updates are shortened to land exactly on token milestones. The loss chunk size is 1,024 and the usual optimizer update consumes 65,536 target tokens.

Full training uses approximately 16.2 GB for FP32 parameters, gradients and AdamW states alone, before activations and temporary buffers. An 8 GB GPU needs an adapted training approach or memory offloading. The GPU startup probe only checks a short BF16 grouped-query FlashAttention forward/backward pass. Measure long-context memory and throughput separately.

Review the displayed hourly compute-unit rate and balance before allocating a paid runtime. Persistent Drive copies and completed downloads protect checkpoints when Colab's temporary runtime ends.

## Verification

With PyTorch and the notebook's dependencies installed:

```bash
python verify_gu_notebook.py
```

The verifier uses small CPU models, synthetic data and mocked Colab operations to check syntax, causal attention, loss gradients, exact milestones, deterministic CPU resume, data-cursor recovery, exports and generation. T4 precision selection and CUDA fallback policy are simulated. Full GPU memory requirements, speed and real network behavior need separate runtime checks.

## Help us learn

Useful contributions include checkpoint comparisons, reproducible failure cases, better explanations, independent evaluations and improvements to the training implementation. Include the checkpoint token count, library versions, hardware, prompt and decoding settings when reporting results. Share weaker outputs alongside successful ones.

Further experiments will compare greedy and sampled generation, expand held-out evaluation, explore additional pretraining and assess instruction tuning separately. Preserve the base checkpoint so others can repeat and extend the work.

## Anewo and attribution

This is **Anewo's first experimental model release**. We are sharing a learning point in an ongoing research and engineering process. Future larger-model experiments depend on available compute and evidence from these evaluations.

The Qwen3-style architecture adapts the original `qwen3_from_scratch.ipynb` design. PyTorch and Hugging Face libraries provide core infrastructure. The SmolLM repository supplies the tokenizer and cosmopedia-v2 supplies training text. Gu-1B weights were initialized randomly.

The associated Hugging Face model repository declares MIT. Review upstream software, tokenizer and dataset attribution and license information when redistributing components. The supplied artwork is included as project branding.
