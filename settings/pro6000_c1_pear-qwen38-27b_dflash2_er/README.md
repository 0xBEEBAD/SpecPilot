# PEAR + DFlash2, SpecPilot F engine + efficient reasoning (lossy), RTX PRO 6000

Candidate: SpecPilot F engine `d5-009-f` with the decode rule `thr0.3+nohes+norep+retry` ("efficient reasoning").
Target: [PEAR](https://github.com/mit-han-lab/pear) (Qwen3.8-27B trained with RL to act as a robot policy), bf16.
Drafter: DFlash2. Hardware: RTX PRO 6000 Blackwell Server (sm_120), one GPU per server. Regime: single request,
multi-turn LIBERO sessions, PEAR's deployment sampling (T=1, top-p 0.95, `reasoning_effort` xhigh).

> [!WARNING]
> **This setting is lossy.** Unlike every other setting in this repository, the efficient-reasoning rule changes what
> the model writes: its reasoning is no longer a sample from PEAR's own distribution. It was validated only on LIBERO
> closed loop (120 paired episodes) at T=1 / top-p 0.95. With the rule switched off (`ER_RULE=lossless`) the engine is
> the lossless F engine.

## What it is

**F engine (`d5-009-f`, lossless).** The SpecPilot PEAR campaign's final configuration: the RTX PRO 6000 Qwen final
(`k-001-d3qpdl`, see [`pro6000_c1_qwen38-27b_dflash2`](../pro6000_c1_qwen38-27b_dflash2)) plus

- lossless T>0 speculative sampling (`SPECPILOT_K001_SAMPLE`: keyed sampling races, so every committed token is the
  target's own draw under T / top-p),
- PEAR proposal-side calibration (drafter tree temperature 1.5, PEAR-ranked drafter vocabulary in `lab/pear_data/`),
- a prefix cache for the multi-turn sessions (mamba radix cache with decode checkpoints, prompt-boundary checkpoints,
  multimodal preprocessing cache),
- two attention kernels for decode and prefill (`SPECPILOT_D2A_KATTN`, `SPECPILOT_D2A_KPREFILL`; summation order only),
- a data-aware Q copy (`SPECPILOT_D3K_GPTQ`: GPTQ-rounded 4/5/6-bit copy of the weights for the proposal-side verify
  pass, 16.18 GB, from per-matrix calibration statistics; the exact bf16 pass still decides every token).

Measured in that campaign (lossless, T=1): +9.5% [+8.8, +10.1] (session set A) and +11.5% [+10.0, +12.9] (set B)
decode speed over the k001 seed on PEAR LIBERO sessions; T=1 rule audit: 0 of 28,877 committed tokens were not the
sampler's own draw.

**Efficient reasoning (`thr0.3+nohes+norep+retry`, lossy).** A hook in the tree-verify accept step
([`files/python/sglang/srt/speculative/specpilot_lossy.py`](files/python/sglang/srt/speculative/specpilot_lossy.py),
called from `dflash_worker_v2.py`; T>0 requests only), configured by a JSON control file (`PEARDR_CTL`) that the
engine re-reads at every request start:

| part | what it does |
|---|---|
| `thr0.3` | accept a drafted token if the target's probability of it (under T=1, top-p 0.95) is at least 0.3, else fall back to the lossless draw |
| `nohes` | do not sample hesitation tokens (Wait / Hmm / Actually / Hold) in rows where the target has an alternative (rows where they hold at least 0.999 of the mass are untouched) |
| `norep` | loop guard: a drafted token that would complete an 8-token sequence already committed in the request (prompt included) is never relaxed; inside a repetition only the target's own draw commits |
| `retry` | in the closed-loop client: a turn that ends without a valid action is regenerated once with the lossless rule |
| (always) | structural tokens (`<|im_end|>`, `</think>`, ... ids 248044-248319) are only committed by the target's lossless draw |

## Results (measured 2026-11-16, LIBERO closed loop)

120 episodes: libero_spatial / libero_object / libero_goal, tasks 0-9, trials 0-1, seeds 1-2. Each episode is paired
with the lossless F engine's episode on the same task, initial state and seed (the recorded lossless baseline run of
the same engine, one day earlier). Robot clock = per-turn inference wall time + native steps / 20 Hz. 95% CIs are
paired bootstrap over episodes.

| | efficient reasoning | lossless F |
|---|---|---|
| success | **81.7%** | 81.7% |
| success difference (pts) | **0.0 [-5.0, +5.0]** (5 episodes succeeded only here, 5 only in the baseline) | |
| robot clock speedup, episodes where both succeeded (93) | **1.87x [1.59, 2.17]** | 1.00x |
| model time speedup, all 120 episodes | 1.49x | 1.00x |
| generated tokens per episode | 15.6k | 21.9k |
| decode speed | 204 tok/s | ~183-200 tok/s |
| turns without an action / lossless retries | 0 of 1386 / 0 | |
| episodes ended by the 131k context limit | 6 | 9 |

- A same-night lossless control (60 episodes, same scripts) ran at 0.95x [0.88, 1.03] of the baseline's clock with the
  same success rate: the machine was slightly slower that night, so 1.87x is if anything conservative (about 1.97x
  against the control, estimated).
- Components alone (120 episodes each, earlier round): `nohes` 1.61x [1.38, 1.88]; `thr0.3` without the loop guard
  1.22x [1.03, 1.45]; both with success equal to the baseline within the CI.
- Long-turn repetition rate 0.010 (lossless 0.007); turns over 8000 tokens fall from 4.6% to 1.9%.
- All aggregates of that run (every arm, including the rejected `nucleus` / `thr0.1` ones):
  [`measured/cl2_summary.json`](measured/cl2_summary.json).

## Limitations

- **Lossy.** Reasoning text differs from what PEAR would sample; success was equal on LIBERO, but the CI (+-5 pts on
  120 episodes) cannot rule out a small drop.
- **LIBERO only** (spatial, object, goal; not LIBERO-long, LIBERO-Plus / Pro, RoboDojo or a real robot).
- **Threshold must not go below 0.3.** Lower thresholds let the drafter's "repeat what was written" proposals through:
  the model loops verbatim and turns end without an action (`nucleus`: 7 of 27 episodes, `thr0.1`: 3 of 60).
  `scripts/rule.py` refuses them unless `--allow-below-0.3`.
- **T=1 / top-p 0.95 only.** The hook acts only on T>0 requests and was measured only at PEAR's deployment sampling.
- Single request per server (the loop guard and the measurements assume batch 1); RTX PRO 6000 (sm_120) only.
- The data-aware Q copy needs external calibration statistics (below); they are not in this repository.

## Prerequisites

| what | how |
|---|---|
| SpecPilot environment | `bash scripts/setup_env.sh` (Python 3.11, torch 2.13 + CUDA 13.0; see the top-level README) |
| PEAR weights (bf16) | [mit-han-lab/Qwen3.8-27B-PEAR](https://huggingface.co/mit-han-lab/Qwen3.8-27B-PEAR), revision `65176e435e5fa80a059ebfafaf8215e6ad00a83e` |
| DFlash2 drafter | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`: the same drafter and directory as the other Qwen settings. PEAR's release points to [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), whose weights and config are byte-identical (same `model.safetensors` sha256) |
| calibration statistics | directory `hess_r_029035503129f133` (content hash `029035503129f133`): 256 per-matrix Hessians `<module>.H.pt` (~94 GB) + `CONTENT_HASH`; at first load the engine computes the Q-copy codes into `<dir>/codes` (~45 GB, so the directory must be writable unless `codes/` is already there). Not published: copy it from the measurement machine, regenerate it (below), or serve without it (`PEAR_CALIB=`) |
| PEAR code + LIBERO | a checkout of [mit-han-lab/pear](https://github.com/mit-han-lab/pear) (measured at commit `f32aa1f`) and its LIBERO simulator env and Mesa: `WORK_DIR=<dir> bash scripts/install.sh libero` and `... mesa` in that checkout |
| GPUs | RTX PRO 6000 Blackwell (96 GB), one per server; `--mem-fraction-static 0.91` |

```bash
bash scripts/download_models.sh pro6000_c1_pear-qwen38-27b_dflash2_er   # -> $XINFER_SHARED/models/{Qwen3.8-27B-PEAR,Qwen3.8-27B-DFlash2}
```

## Reproduce

All scripts are in [`scripts/`](scripts); paths come from the environment (defaults in brackets):

| variable | used by | meaning |
|---|---|---|
| `SPECPILOT_DATA` [`~/specpilot_data`] | all | data root of this repository (`shared/`: venv, models, caches; `work/`: engines, logs, episodes) |
| `PEAR_MODEL` [`$XINFER_SHARED/models/Qwen3.8-27B-PEAR`] | serve | target checkpoint |
| `PEAR_DRAFT` [`$XINFER_SHARED/models/Qwen3.8-27B-DFlash2`] | serve | drafter |
| `PEAR_CALIB` [`$XINFER_SHARED/pear_calib/hess_r_029035503129f133`] | serve | calibration directory; empty = no data-aware Q copy (lossless, a few % slower, not the measured configuration) |
| `ER_RULE` [`er`] | serve | rule at start (`er`, `lossless`, or any `rule.py` arm) |
| `ER_CTL` / `ER_CTL_DIR` [`$SPECPILOT_WORK/pear_er/ctl/port_<port>.json`] | serve / closed loop | control file per server |
| `GPU`, `PORT` [0, 29500 + 10 * GPU] | serve | or as positional arguments |
| `GPUS`, `BASE_PORT` [0, 29500] | reproduce | one server per listed GPU |
| `PEAR_REPO` (required), `PEAR_WORK_DIR` | episodes | PEAR checkout; its install dir (gives `LIBERO_PYTHON`, `LIBERO_MESA_DIR`, `LIBERO_CONFIG_PATH`, which can also be set directly) |
| `ER_OUT` [`$SPECPILOT_WORK/pear_er/episodes/<name>`] | closed loop | episode output |
| `DRY_RUN=1` | serve, reproduce | build the engine tree, print environment, command and job list; start nothing |

```bash
# (a) serve: builds the engine (stock engine + files/) once, writes the control file, starts and waits for the server
PEAR_CALIB=/path/to/hess_r_029035503129f133 bash settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/serve.sh 0
python3 settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/rule.py lossless --out $SPECPILOT_WORK/pear_er/ctl/port_29500.json  # switch rule live
bash scripts/stop.sh 29500

# (b) + (c) paired closed loop on GPUs 0-3 (both arms on every server, interleaved per episode), then the summary
GPUS=0,1,2,3 PEAR_REPO=/path/to/pear PEAR_WORK_DIR=/path/to/pear_work PEAR_CALIB=/path/to/hess_r_029035503129f133 \
  bash settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/reproduce.sh            # add --limit 6 for a smoke run

# or step by step, against servers already running on ports 29500,29510:
python3 settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/closed_loop.py --ports 29500,29510 --arms lossless,er
python3 settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/summarize.py $SPECPILOT_WORK/pear_er/episodes/er_eval
```

- `closed_loop.py` runs one episode at a time per server and sets the server's control file to the episode's arm
  before it starts; finished episodes are skipped, so a run can be resumed. Default grid = the measured 120 episodes
  per arm (about 8 GPU-hours for both arms).
- `run_episode.py` is PEAR's own LIBERO agent loop (system prompt, JSON action protocol, `LiberoEnv`) with the OpenAI
  API of the server as the backend; it runs in PEAR's LIBERO env via `run_episode.sh`.
- `summarize.py` prints, per arm against the lossless arm: success and its paired difference with CI, tokens per
  episode, robot clock and its speedup on episodes where both succeeded with CI, model-time speedup, decode tok/s,
  end reasons, turns without an action and lossless retries; `--json` writes the same.
- Differences from the measurement: there the baseline was the recorded lossless run of the previous night (paired by
  episode, plus a 60-episode same-night lossless control); here the lossless arm runs in the same session, which is
  the cleaner pairing. The servers there used file-based torch.distributed init and a pre-warmed kernel cache; the
  first boot here compiles kernels (run one episode before measuring).

### Calibration statistics (regeneration, approximate)

The measured statistics came from prefill inputs of 8 PEAR LIBERO sessions disjoint from all evaluation episodes
(stride 2, 84,775 rows per matrix). To make a new set with these scripts:

```bash
# 1. capture prefill input rows of the Q-copied linears (no calibration, no prefix cache, eager prefill)
PEAR_CALIB= RADIX=0 ER_RULE=lossless \
SPECPILOT_D3K_CAPTURE=/path/to/capture:mlp.gate_up_proj,mlp.down_proj,linear_attn.in_proj_qkvz,linear_attn.out_proj,.qkv_proj,.o_proj \
  bash settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/serve.sh 0 29500 --cuda-graph-backend-prefill=disabled
python3 settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/closed_loop.py --ports 29500 --arms lossless \
  --name calib --trials 2 --tasks 0-2 --seeds 0 --suites libero_spatial,libero_goal   # initial states outside the eval grid
bash scripts/stop.sh 29500
# 2. Hessians (one GPU): H = X^T X / n per matrix
python3 settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/calib_hess.py --calib /path/to/capture --out /path/to/hess_new --heldout-from 1000000
sha1sum /path/to/hess_new/*.H.pt | sha1sum | cut -c1-16 > /path/to/hess_new/CONTENT_HASH   # any label; the engine only needs the file to exist
```

A new set gives different Q-copy codes: the engine stays lossless (the exact pass decides), but its speed can differ
slightly from the measurement. The capture can take a lot of disk (bf16 rows of every captured linear).

## What was and was not executed for this commit

No GPU was used while preparing it (all GPUs were busy), so **nothing here was served or run end to end from this
repository.** Executed on CPU:

- `scripts/build_engine.py pro6000_c1_pear-qwen38-27b_dflash2_er final` builds a tree identical, file by file, to the
  engine that produced the measurements (tree sha `db7ebb6b468b67ca`);
- `serve.sh` / `reproduce.sh` with `DRY_RUN=1`: the server environment equals the measured servers' `SPECPILOT_*` /
  `PEARDR_*` environment except `SPECPILOT_BOOT_CONCURRENCY` (boot scheduling) and `SPECPILOT_BLOCK_SIZE` (this
  repository's name for the verify width 32 that the measurement passed as `--speculative-num-draft-tokens 32`);
- `tests/test_norep.py` (loop guard, CPU, needs torch) and `tests/test_rule.py` (control configs; `rule.py` matches the
  measurement's arm configs) pass;
- `summarize.py` on the original episode records (baseline = the recorded lossless run): 1.87x [1.59, 2.19] on the 107
  pairs with complete records (the published 120-pair numbers also count context-limit failures, whose logs are not in
  this layout);
- `closed_loop.py --dry-run` (job grid and control configs); `run_episode.sh --help` in PEAR's LIBERO env (imports
  PEAR's protocol and `LiberoEnv`);
- `bash -n` on the shell scripts, `py_compile` on the Python files.

Not executed: serving, the closed loop, the calibration regeneration, `download_models.sh`.

## Files

- `config.json`: models and revisions, server flags, the final's switches (`final.env`), path-dependent switches
  (`final.path_env`, filled in by `scripts/serve.sh`), the rule, the measured numbers
- `files/`: engine files the final adds or changes over the stock engine (85: the 54 of `k-001-d3qpdl`, 11 of them
  changed, plus 31 new, among them the hook `specpilot_lossy.py`, and `lab/pear_data/`), copied over the stock engine by
  `scripts/build_engine.py`; SGLang-derived files keep their Apache-2.0 headers (repository licence: Apache-2.0)
- `scripts/`: `serve.sh`, `rule.py`, `closed_loop.py`, `run_episode.py` + `run_episode.sh`, `summarize.py`,
  `reproduce.sh`, `calib_hess.py`
- `tests/`: `test_norep.py`, `test_rule.py`
- `measured/cl2_summary.json`: the closed-loop summary of the measurement (aggregates only)
