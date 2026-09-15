# Slovenian NER optimization history

## Method (2026-09-15)

RTX 4090, Windows, Python 3.14, native FP32 Slovenian standard NER model.
Isolated editable environment: `C:/GitHub/semora/artifacts/ner-benchmark/.venv`.
Baseline CLASSLA commit: `265ae9b`; Semora: `214c480`.
Each incremental run uses the same first 2,000 articles of the seed-42 uniform
10,000-article sample (sample SHA256
`c98d779ac2f59f726229223dd7901f09d4b038f372b9eb38283b0763e5a78976`).
Model SHA256: `30e9efafdf991e60a31b8037f72e90ed4b7b3c942cc5a32fc47eeb280bbffa22`.
Sentence batch size 16, outer article batch size 32 unless stated otherwise.
Times exclude model loading/warm-up but include sample reads, extraction,
span validation and indexed SQLite writes. These are short single runs, not
statistically established speedups or full-production indexing measurements.
NER time includes CPU input preparation and decoding, not just GPU compute.

Artifacts live below `C:/GitHub/semora/artifacts/ner-benchmark/` (not committed).
Each run retains `metrics.json` and `predictions.sqlite`. Equivalence means a
bidirectional comparison of complete per-article entity JSON, including labels,
text and character offsets; entity counts alone are not sufficient.

| Step / artifact directory | Wall seconds | Articles/s | Tokenization s | NER s | Exact baseline equivalence |
|---|---:|---:|---:|---:|---|
| Baseline: `ner-optimization-baseline` | 60.508 | 33.05 | 18.234 | 41.310 | Reference |
| 1: `ner-optimization-step1` | 37.591 | 53.20 | 17.400 | 19.259 | 2,000/2,000 articles |
| 2: `ner-optimization-step2` | 34.940 | 57.24 | 17.006 | 17.009 | 2,000/2,000 articles |

## Step 1: skip training loss during prediction

Add `compute_loss=True` to the NER model, preserving training/evaluation-loss
callers; prediction explicitly disables loss and uses `torch.inference_mode()`.
Keep the learned CRF transitions for Viterbi decoding. The benchmark adapter
already disabled gradients, so its gain is not attributable to that decorator.
Observed wall-time reduction: 37.9%; throughput: 1.61x baseline.
Tests: mock inference guard; real-model identical logits/transitions with and
without loss; finite evaluation loss; training backward gradients. Both pass.

## Step 2: inference inputs and packed tensors

Skip target-tag preparation during prediction and skip character-language-model
sequence construction/transfers when `charlm=false`. Preserve the actual word
character encoder and the charLM input path for models that need it. During eval,
avoid two pad/dropout/repack round trips; retain training dropout behavior.
Observed wall-time reduction versus step 1: 7.1%; throughput: 1.73x baseline.
Three tests pass, including charLM input retention, training backward and
identical predictions with full versus inference-only inputs.

## Reproduce a run

```powershell
$py = 'C:/GitHub/semora/artifacts/ner-benchmark/.venv/Scripts/python.exe'
& $py -m semora.diagnostics.ner run `
  --sample C:/GitHub/semora/artifacts/ner-benchmark/sample.sqlite `
  --adapter classla --device cuda --batch-size 16 --batch-articles 32 `
  --limit 2000 --output C:/GitHub/semora/artifacts/ner-benchmark/NEW_RUN
```

Use a new output directory each time. Stop other GPU workloads before comparing.
