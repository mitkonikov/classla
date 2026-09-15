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
| 3: `ner-optimization-step3/batch-16-repeat-1` | 31.537 | 63.42 | 16.684 | 13.941 | 2,000/2,000 articles |
| 3: `ner-optimization-step3/batch-32-repeat-1` | 32.463 | 61.61 | 16.984 | 14.511 | 2,000/2,000 articles |
| 3: `ner-optimization-step3/batch-64-repeat-1` | 38.863 | 51.46 | 19.439 | 17.900 | 2,000/2,000 articles |
| 4: `ner-optimization-step4-sequential` | 32.239 | 62.04 | 16.835 | 14.434 | 2,000/2,000 articles |
| 4: `ner-optimization-step4-threads` | 32.867 | 60.85 | 31.613 | 26.590 | 2,000/2,000 articles |
| 4: `ner-optimization-step4-workers2` | 17.393 | 114.99 | 9.298 | 17.099 | 2,000/2,000 articles |
| 4: `ner-optimization-step4-workers4` | 17.544 | 114.00 | 6.930 | 15.925 | 2,000/2,000 articles |
| 5: `ner-optimization-step5` | 13.737 | 145.60 | 9.425 | 13.445 | 2,000/2,000 articles |
| Original-order control: `ner-optimization-conservative2k` | 34.255 | 58.39 | 15.800 | 33.840 | 2,000/2,000 articles |
| 6: `ner-optimization-step6` (original order) | 32.508 | 61.52 | 16.002 | 32.094 | 2,000/2,000 articles |

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

## Step 3: length-grouped sentence batches

Sort sentences across the bounded outer article batch before forming model
batches, then restore document order after prediction. Training order is
unchanged. Set `ner_sort_during_eval=False` in the CLASSLA pipeline to disable.
All three configurations match baseline entities exactly. Batch 16 remains
fastest in this short sweep: 9.7% lower wall time than step 2, 1.92x baseline
throughput. Four regression tests pass, including order restoration and empty
input. Batch-size changes can alter floating-point arithmetic; exact agreement
here is a measured sample result, not a universal bitwise guarantee.

## Step 4: bounded CPU/GPU/write overlap (Semora)

The implementation is in Semora's `diagnostics/ner/staging.py` and adapter,
reusing CLASSLA's existing `process_tokenized` entry point. One preparation
thread reads sample rows, optionally dispatches CPU-only tokenization to persistent
spawned workers, reconstructs documents and validates article-relative offsets.
One inference thread owns the GPU model; the calling thread validates and writes
results. Pipeline depth bounds queued work; no CUDA tensors/models cross IPC.
No new production index is built. Threads/pools close on completion and failures.

Sequential control uses depth 1; all overlapped runs use depth 3. Thread-only
overlap provides no gain here. Two CPU tokenizer workers achieve 1.85x the new
sequential control and 3.48x the original baseline; four workers are not faster
in this short sample. These observations are consistent with CPU contention but
do not isolate the GIL as the sole cause. Stage timings overlap and include
contention/waiting, so must not be summed to infer wall time or GPU utilization.
Parent RSS excludes tokenizer-worker memory. Warm-up uses two articles; additional
lazy worker startup, if any, remains included in measured wall time.

Thirteen Semora tests pass (two unrelated opt-in model tests skipped); type check
and lint pass. Tests cover stable order, a single inference owner, exception
propagation and early-close cleanup. All four real-model runs preserve baseline
entity JSON exactly. New metrics fingerprint editable CLASSLA NER code, and
summary grouping includes pipeline depth to avoid mixing different configurations.

Use `--pipeline-depth 3 --tokenizer-workers 2` to enable this path. Defaults remain
sequential for compatibility and explicit timing comparisons.

## Step 5: batched CPU Viterbi decoding

A 512-article sequential cProfile run (`ner-step4-profile.prof`) found 7,040
scalar Viterbi calls taking 1.518 s of 5.324 s inside NERProcessor (including
warm-up). The profiled run is diagnostic only; instrumentation overhead makes
its 15.325 s extraction wall time unsuitable for throughput comparison.

Vectorize the NumPy recurrence across independent sentences, keeping scalar
per-sentence backtrace, first-index argmax tie breaking and original score dtype.
The existing scalar decoder remains available and unchanged. No new dependency
or approximate decoder is introduced. Nine NER tests pass, covering random scores,
ties, float32/float64, unequal/one-token lengths, empty batches, real-model loss
and training. Eleven earlier POS/lemma regression tests also pass.

At depth 3 / two tokenizer workers, observed wall time falls another 21.0% versus
step 4 workers2, reaching 4.40x original baseline throughput. Complete entity JSON
matches for all 2,000 articles. A final larger-sample repeat check follows below.

## Larger-sample findings and conservative defaults

The first full 10,000-article run with length grouping (`ner-optimization-final10k/
batch-16-repeat-1`) took 85.326 s, 117.20 articles/s. It matched 9,998/10,000
historical baseline article outputs: one PER mention and one ORG mention were
absent. Offsets of retained entities were unchanged. These were outside the
2,000-article tuning prefix, so the earlier exact comparisons remain valid.

Rerunning the two original 32-article batches isolated the change to sentence
grouping: both scalar and batched decoders differ with grouping enabled; both
match the historical baseline with original sentence order. This is consistent
with batch-dependent numerical effects, not a batched decoder discrepancy.
Consequently grouping is now OPT-IN: CLASSLA `ner_sort_during_eval=True`, or
Semora `--sentence-order length`. Default `input` order preserves original batches.
The length-grouping optimization remains available, but is not represented as
universally prediction-identical. Final conservative validation is recorded below.

The second full-sample repeat failed after 3,232 committed articles due to Windows
denying an atomic `metrics.tmp` replacement (`WinError 5`). It is retained as
failed, never counted as a fast run. Semora now retries this transient permission
error at most five times (1.55 s total backoff), still raising persistent failures;
tests cover both outcomes. The third grouped repeat completed in 132.678 s
(75.37 articles/s), with the same two changed articles. This substantial timing
variation is another reason not to advertise the fastest short grouped run as a
stable full-corpus speedup. Both successful grouped passes produced 145,077
entities, versus 145,079 in the historical baseline.

## Step 6: avoid decoding completed/padded sentence rows

The original-order control had mixed sentence lengths. The batched decoder now
sorts only already-computed score rows internally and shrinks its active prefix
as sentences end, then restores output order. Neural inference batches are NOT
reordered. Regression tests include one 200-token sentence mixed with 1-3-token
sentences, random scores, exact ties and both dtypes. All nine NER tests pass.

The 2,000-article run matches every baseline entity and takes 32.508 s versus
34.255 s for the immediate original-order control (5.1% lower wall time). Its
throughput is 1.86x the initial baseline. This is slower than the optional grouped
configuration; retaining original neural batch composition has a measurable cost.
The unfinished `ner-optimization-conservative10k/batch-16-repeat-1` was explicitly
stopped before applying this change and marked failed; it is not a timing result.

## Final conservative validation (three complete 10,000-article passes)

CLASSLA code: `6bb8c96`; Semora staging/defaults: `385a9bf` (summary columns:
`0c0a4ec`). Settings: sentence batch 16, outer batch 32, pipeline depth 3,
two tokenizer workers, original/input sentence order. The stricter pipeline
bound includes the batch being written. No competing benchmark models ran.

| Artifact under `ner-optimization-verified10k/` | Wall s | Articles/s | Tokenization s | NER s | Exact baseline equivalence |
|---|---:|---:|---:|---:|---|
| `batch-16-repeat-1` | 160.705 | 62.226 | 80.669 | 159.319 | 10,000/10,000 |
| `batch-16-repeat-2` | 161.660 | 61.858 | 80.907 | 160.331 | 10,000/10,000 |
| `batch-16-repeat-3` | 161.680 | 61.851 | 80.840 | 160.397 | 10,000/10,000 |

Mean wall time: **161.348 +/- 0.557 s** (sample standard deviation).
Mean throughput: **61.978 +/- 0.214 articles/s**. Each pass contains exactly the
same 145,079 entities as `classla-aligned-sweep/batch-16-repeat-1`, verified by
bidirectional full per-article JSON comparison (zero missing or differing rows).
Peak allocated CUDA memory was 113,893,376 bytes in each final pass; parent RSS
does not represent total RAM including tokenizer workers.

The earlier full-sample baseline took 272.435 s (36.706 articles/s): approximately
**1.69x throughput / 40.8% lower wall time**. That baseline was measured earlier,
not as part of these three repeats. The fresh 2,000-article comparison gives
1.86x throughput; do not mix sample sizes when comparing raw durations.
These are offline sample benchmarks, not a full-corpus indexing estimate or an
hour-long stability certification. No production database/index was modified.

Machine-readable repeat summary: `ner-optimization-final-summary/summary.csv`
and `summary.json`, plus `statuses.csv`. Individual metrics/prediction databases,
profiling results and failed/interrupted outputs are retained for auditing.
Final focused checks: 9 NER tests (including real model and backward), 11 prior
POS/lemma regression tests, 17 Semora tests; Semora lint and type checking pass.

## Round 2: CPU/CUDA timeline and inner NER pipeline (2026-09-15)

The new 512-article sequential CPU profile (`ner-round2-cpu.prof`) attributes
0.720 s to 894 input-tensor constructions, 0.649 s to 17 DataLoader initializations,
and 0.346 s to 447 batched decodes, out of 5.228 s inside NERProcessor including
warm-up. These instrumented times are diagnostic, not throughput measurements.
A bounded 32-article CPU/CUDA trace (`ner-round2-timeline.json`) is generated by
Semora's new `diagnostics.ner.profile_classla` module. It labels vocabulary mapping,
CPU tensor creation, input transfers, neural inference and CPU decoding. CUDA
annotation totals can overlap; do not sum annotation and kernel timings.

### Bulk input construction

NER's two-dimensional input arrays now use one NumPy allocation and one shared
Torch view instead of creating a Torch tensor for every row. Padding, int64 dtype
and contiguity match the original helper. A 400-row microbenchmark, 100 calls,
best of three repetitions, took 0.202103 s with the original Torch loop versus
0.020696 s with NumPy (9.77x construction throughput). Ten NER tests pass.

| 2,000-article trial | Wall s | Articles/s | Tokenization s | NER s | Exact baseline equivalence |
|---|---:|---:|---:|---:|---|
| `ner-round2-baseline` | 16.054 | 124.58 | 9.105 | 15.752 | Reference |
| `ner-round2-bulk` | 22.908 | 87.31 | 16.466 | 22.519 | 2,000/2,000 |

The single end-to-end bulk trial was slower despite the microbenchmark gain;
tokenization time also changed substantially. Do not attribute that whole-run
difference to tensor construction or compare these rates directly to round 1.
Subsequent controlled configuration trials are recorded below.

### Inner preparation/decoding and optional Numba trial

Split prediction into CPU-score production and CPU decoding. The optional inner
pipeline prepares CPU tensors ahead and decodes earlier scores while the caller
runs the GPU model. Each preparation/decoding queue is bounded by inner depth;
the currently executing batch is additional. One thread owns each CPU stage and
GPU inference stays on the caller. Exceptions and early close shut down both
executors. Default inner depth is 1, retaining serial execution.

Optional `classla[ner-jit]` uses Numba (tested 0.67.0 / llvmlite 0.49.0) for Viterbi.
Only the isolated benchmark environment was extended; existing Torch/NumPy and
Nora's environment were not changed. The kernel uses `nogil=True`, caching, no
fast-math, original precision and first-index tie breaking. NumPy remains the
default and handles non-finite/unusual-dtype inputs. Compilation is outside the
steady-state timer in model warm-up; warm-up durations are retained in metrics.

| 2,000-article trial | Wall s | Articles/s | Tokenization s | NER s | Exact baseline equivalence |
|---|---:|---:|---:|---:|---|
| `ner-round2-numpy-depth1` | 23.213 | 86.16 | 16.394 | 22.815 | 2,000/2,000 |
| `ner-round2-numpy-depth2` | 21.262 | 94.06 | 16.317 | 20.867 | 2,000/2,000 |
| `ner-round2-numpy-depth3` | 14.745 | 135.64 | 11.334 | 14.396 | 2,000/2,000 |
| `ner-round2-numba-depth2` | 12.363 | 161.77 | 10.053 | 12.044 | 2,000/2,000 |
| `ner-round2-numba-depth1` | 11.512 | 173.73 | 10.049 | 11.215 | 2,000/2,000 |

All runs keep outer depth 3, two tokenizer workers and original sentence order.
Twenty NER tests pass: real-model/training, input equivalence, pipeline ownership/
order/failure cleanup, and Numba-versus-NumPy random/tied/uneven/non-finite cases.
Threading is not assumed faster: Numba depth 1 won this initial sweep. Alternating
controls and a full-sample validation follow before recommending settings.

Pinned input copies are deferred: the bounded trace recorded only 0.184 ms of
pageable HtoD copy device time across 125 events, versus 22.174 ms of cuDNN RNN
device time. This is not a claim that synchronization is free; it means copy
bandwidth is not the first target. No CUDA graph/torch.compile change was made.

## Reproduce a run

```powershell
$py = 'C:/GitHub/semora/artifacts/ner-benchmark/.venv/Scripts/python.exe'
& $py -m semora.diagnostics.ner run `
  --sample C:/GitHub/semora/artifacts/ner-benchmark/sample.sqlite `
  --adapter classla --device cuda --batch-size 16 --batch-articles 32 `
  --pipeline-depth 3 --tokenizer-workers 2 --sentence-order input `
  --output C:/GitHub/semora/artifacts/ner-benchmark/NEW_RUN
```

Use a new output directory each time. Stop other GPU workloads before comparing.
Add `--limit 2000` for the short tuning sample. Optional length grouping is a
speed/strict-reproducibility trade-off, not the default; changing neural batch
size/composition can also change near-tied model decisions.
