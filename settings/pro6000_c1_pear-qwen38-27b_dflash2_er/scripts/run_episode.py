"""One closed-loop PEAR LIBERO episode against the OpenAI-compatible server of scripts/serve.sh.
(From the pear-drafter experiments' run_episode2.py; changes: the PEAR checkout comes from $PEAR_REPO, docs.)

Reproduces PEAR's evaluation loop (pear/agent_loop/agent_loop.py PearAgentLoop + pear/envs/tool.py EnvironmentTool +
pear.envs.libero.env.LiberoEnv) with only the inference backend swapped:
  * env: pear.envs.libero.env.LiberoEnv(suite, task_id, trial_id) used in-process exactly as its simulator process
    does (stateless rollout: replay native actions, carry out one action);
  * system prompt / user turns / reminder / action check: PEAR's own JsonBlockProtocol + EnvironmentTool texts;
  * conversation: full history, earlier assistant turns kept with their reasoning (chat template with
    enable_thinking, reasoning_effort=xhigh, preserve_thinking), frames as lossless PNG;
  * sampling: T=1, top_p 0.95, top_k -1 (PEAR's deployment sampling; the only setting the rule was measured at);
  * episode ends: env done (success/fail), parse failure, or 256000 response-token budget.
Records per turn: wall-clock request start/first token/last token, token counts, streamed text chunks with timestamps,
the action, native steps executed, env time. Run with the LIBERO env's python (run_episode.sh does that).
Safety net (--retry-ctl-file): a turn that ends without a valid action is regenerated once with the lossless rule."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

PEAR = Path(os.environ.get("PEAR_REPO") or sys.exit("set PEAR_REPO to a checkout of github.com/mit-han-lab/pear"))
sys.path.insert(0, str(PEAR))
from pear.agent_loop.protocol import InvalidAction, JsonBlockProtocol  # noqa: E402
from pear.envs.libero.env import LiberoEnv  # noqa: E402

SYSTEM_TEMPLATE = (PEAR / "pear/envs/system_prompt.md").read_text()
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True, "reasoning_effort": "xhigh", "preserve_thinking": True}
RESPONSE_BUDGET = 256000

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--backend", required=True)  # label only
ap.add_argument("--suite", required=True)
ap.add_argument("--task", type=int, required=True)
ap.add_argument("--trial", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)  # per-request seed = seed*1000 + turn
ap.add_argument("--out", required=True)
ap.add_argument("--max-turn-tokens", type=int, default=32768)
ap.add_argument("--max-wall", type=float, default=3600.0)  # give up on an episode after this much inference time
ap.add_argument("--rid-prefix", default="")  # request ids "<prefix>|<turn>" for the engine's lossy stats
# safety net: when a turn ends without a valid action (not because the token budget ran out), switch the
# server's lossy control file to lossless, regenerate the same turn once, then restore the arm's control file. The
# turn's infer_s / completion_tokens count both attempts; the history keeps only the regenerated answer.
ap.add_argument("--retry-ctl-file", default="")
ap.add_argument("--retry-arm-json", default="")      # the arm's control-file content (restored after the retry)
ap.add_argument("--force-retry-turn", type=int, default=-1)   # testing only: treat this turn's first attempt as failed
a = ap.parse_args()

out = Path(a.out)
(out / "frames").mkdir(parents=True, exist_ok=True)
turns_f = open(out / "turns.jsonl", "w")
chunks_f = open(out / "chunks.jsonl", "w")


def png_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def observation(result, k: int) -> list:
    text = JsonBlockProtocol.turn_text(result["state"], REMINDER)
    blocks = [{"type": "text", "text": text}]
    for name, image in zip(LiberoEnv.CAMERAS, result["images"]):
        image.save(out / "frames" / f"obs_{k:03d}_{name}.png")
        blocks.append({"type": "text", "text": f"[{name}]"})
        blocks.append({"type": "image_url", "image_url": {"url": png_url(image)}})
    return blocks


def chat(messages, max_tokens, seed, rid=None):
    body = {"model": "qwen3_8_27b", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 1.0, "top_p": 0.95, "top_k": -1,
            "seed": seed, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS}
    if rid:
        body["rid"] = rid
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    tf = tl = None
    usage, finish = None, None
    matched = None
    reasoning, content, chunks = [], [], []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):
                ch = d["choices"][0]
                delta = ch.get("delta", {})
                c = delta.get("content")
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if c or rc:
                    tl = time.time()
                    tf = tf or tl
                    if rc:
                        reasoning.append(rc)
                        chunks.append([round(tl - t0, 4), rc])
                    if c:
                        content.append(c)
                        chunks.append([round(tl - t0, 4), c])
                finish = ch.get("finish_reason") or finish
                if ch.get("matched_stop") is not None:
                    matched = ch.get("matched_stop")
    t_end = time.time()
    R, C = "".join(reasoning), "".join(content)
    if not R and "</think>" in C:  # server without a reasoning parser: split the raw text
        R, _, C = C.rpartition("</think>")
    return dict(t0=t0, tf=tf, tl=tl, t_end=t_end, reasoning=R, content=C, chunks=chunks, usage=usage or {},
                finish=finish, matched=matched)


env = LiberoEnv(a.suite, a.task, a.trial)
te = time.time()
result = env.rollout([], None)
env_s0 = time.time() - te
json.dump(result["state"], open(out / "init_state.json", "w"))  # turn-0 state (not in turns.jsonl)
history: list = []
system = JsonBlockProtocol.system_prompt(SYSTEM_TEMPLATE, LiberoEnv.environment_prompt(result), result["instruction"])
messages = [{"role": "system", "content": system}, {"role": "user", "content": observation(result, 0)}]
meta = dict(backend=a.backend, port=a.port, suite=a.suite, task=a.task, trial=a.trial, seed=a.seed,
            instruction=result["instruction"], start=time.time(), env_open_s=env_s0)
def set_ctl(cfg):
    tmp = a.retry_ctl_file + ".tmp"
    json.dump(cfg, open(tmp, "w")); os.replace(tmp, a.retry_ctl_file); time.sleep(0.1)


def parse_action(text, row):
    blocks = JsonBlockProtocol.JSON_BLOCK_PATTERN.findall(text)
    if blocks:
        try:
            action = json.loads(blocks[-1])
            JsonBlockProtocol.check_action(action, LiberoEnv.SCHEMA)
            return action
        except (json.JSONDecodeError, InvalidAction) as exc:
            row["parse_error"] = repr(exc)
    return None


used = 0
n_retry = 0
end_reason, success, done = None, False, False
infer_total = 0.0
k = 0
while True:
    max_tokens = min(RESPONSE_BUDGET - used, a.max_turn_tokens)
    rid = f"{a.rid_prefix}|{k}|{os.urandom(4).hex()}" if a.rid_prefix else None
    r = chat(messages, max_tokens, a.seed * 1000 + k, rid)
    ct = int(r["usage"].get("completion_tokens") or 0)
    pt = int(r["usage"].get("prompt_tokens") or 0)
    cached = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    used += ct
    infer_s = r["t_end"] - r["t0"]
    infer_total += infer_s
    for t, s in r["chunks"]:
        chunks_f.write(json.dumps([k, t, s]) + "\n")
    row = dict(turn=k, t0=r["t0"], ttft_s=(r["tf"] or r["t_end"]) - r["t0"], infer_s=infer_s,
               decode_s=((r["tl"] or r["t_end"]) - (r["tf"] or r["t0"])), completion_tokens=ct, prompt_tokens=pt,
               cached_tokens=cached, finish=r["finish"], reasoning=r["reasoning"], content=r["content"], rid=rid,
               matched_stop=r.get("matched"))
    action = parse_action(r["content"], row)
    if k == a.force_retry_turn:
        action = None
    if action is None and a.retry_ctl_file and used < RESPONSE_BUDGET:
        arm_cfg = json.loads(a.retry_arm_json)
        set_ctl({"mode": "off", "stats": arm_cfg.get("stats")})
        try:
            rid2 = f"{a.rid_prefix}|{k}r|{os.urandom(4).hex()}" if a.rid_prefix else None
            r2 = chat(messages, min(RESPONSE_BUDGET - used, a.max_turn_tokens), a.seed * 1000 + k + 500, rid2)
        finally:
            set_ctl(arm_cfg)
        n_retry += 1
        ct2 = int(r2["usage"].get("completion_tokens") or 0)
        used += ct2
        infer2 = r2["t_end"] - r2["t0"]
        infer_total += infer2
        for t, s_ in r2["chunks"]:
            chunks_f.write(json.dumps([k, round(infer_s + t, 4), s_]) + "\n")
        row["retry"] = dict(first_tokens=ct, first_infer_s=infer_s, first_decode_s=row["decode_s"], first_finish=r["finish"],
                            first_matched_stop=r.get("matched"), first_reasoning=r["reasoning"], first_content=r["content"],
                            first_parse_error=row.pop("parse_error", None), rid=rid2, forced=(k == a.force_retry_turn))
        row.update(infer_s=infer_s + infer2, decode_s=row["decode_s"] + ((r2["tl"] or r2["t_end"]) - (r2["tf"] or r2["t0"])),
                   ttft_s=row["ttft_s"], completion_tokens=ct + ct2, finish=r2["finish"], reasoning=r2["reasoning"],
                   content=r2["content"], matched_stop=r2.get("matched"), answer_tokens=ct2)
        ct = ct + ct2; infer_s = infer_s + infer2
        r = r2
        action = parse_action(r["content"], row)
    if action is None:
        end_reason = "budget" if used >= RESPONSE_BUDGET else "parse"
        turns_f.write(json.dumps(row) + "\n"); turns_f.flush()
        break
    te = time.time()
    result = env.rollout(history, action)
    row["env_s"] = time.time() - te
    history.extend(result["actions"])
    done = bool(result["done"])
    success = done and sum(float(x) for x in result["rewards"]) > 0.0
    row.update(action=action, executed=result["executed"], outcome=result["outcome"], done=done, success=success,
               steps_total=len(history), state=result["state"])
    turns_f.write(json.dumps(row) + "\n"); turns_f.flush()
    print(f"[{a.backend} {a.suite}/{a.task}/{a.trial} s{a.seed}] turn {k}: {ct} tok in {infer_s:.1f}s "
          f"({ct / max(row['decode_s'], 1e-9):.1f} tok/s) exec {result['executed']} {result['outcome']} done={done} "
          f"succ={success}", flush=True)
    if done:
        end_reason = "env"
        break
    if used >= RESPONSE_BUDGET:
        end_reason = "budget"
        break
    if infer_total > a.max_wall:
        end_reason = "wall"
        break
    messages.append({"role": "assistant", "content": r["content"], "reasoning": r["reasoning"],
                     "reasoning_content": r["reasoning"]})
    k += 1
    messages.append({"role": "user", "content": observation(result, k)})

meta.update(end=time.time(), end_reason=end_reason, success=success, done=done, turns=k + 1, tokens=used, retries=n_retry,
            infer_s=infer_total, native_steps=len(history), actions=history)
json.dump(meta, open(out / "episode.json", "w"))
print(f"EPISODE_END {a.backend} {a.suite}/{a.task}/{a.trial} s{a.seed} success={success} reason={end_reason} "
      f"turns={k + 1} tokens={used} infer={infer_total:.1f}s steps={len(history)}", flush=True)
