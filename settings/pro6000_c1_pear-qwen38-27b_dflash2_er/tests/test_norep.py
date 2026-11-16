"""CPU unit test of the loop guard (norep) in the PEARDR_LOSSY hook of this setting's engine files (no sglang import: the
sampler module is stubbed). Run: python3 settings/pro6000_c1_pear-qwen38-27b_dflash2_er/tests/test_norep.py (needs torch)."""
import importlib.util, json, os, sys, tempfile, types
import torch

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files", "python", "sglang", "srt", "speculative",
                    "specpilot_lossy.py")
ctl = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump({"mode": "nucleus", "norep": 4}, ctl); ctl.close()
os.environ["PEARDR_CTL"] = ctl.name
stub = types.ModuleType("sglang.srt.speculative.specpilot_k001_sample")
STATE = {}
stub.tree_target_probs = lambda logits, info, bs, n_nodes, draft_input: STATE["probs"]
stub.request_seeds = lambda reqs, dev: torch.zeros(1, dtype=torch.int64)
stub.keyed_sample = lambda probs, seeds, positions, salt=0: STATE["draw"]
stub.q_salt = lambda: 1
stub.SALT_EXACT = 0
for name in ["sglang", "sglang.srt", "sglang.srt.speculative"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["sglang.srt.speculative"].specpilot_k001_sample = stub
sys.modules["sglang.srt.speculative.specpilot_k001_sample"] = stub
spec = importlib.util.spec_from_file_location("specpilot_lossy", HOOK)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


class Req:
    rid = "r1"


reqs = [Req()]
H.begin_extend(reqs)
assert H.active() and H._CFG["norep"] == 4, H._CFG
V = 200
ok_all = True


def check(name, cond):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name)
    ok_all &= bool(cond)


def commit(tokens, pad=3):
    """one exact pass that commits `tokens` (out is padded with garbage beyond commit)"""
    out = torch.tensor(list(tokens) + [7] * pad, dtype=torch.int64)
    acc = {"commit_lens": torch.tensor([len(tokens)]), "out_tokens": out.unsqueeze(0),
           "path": torch.zeros(1, 8, dtype=torch.int64)}
    H._LAST["draw"] = None
    H.after_accept(reqs, acc, torch.zeros(1, 8, dtype=torch.int64), q_pass=False)


# committed so far: 10 11 12 13 14 15 | 10 11 12   (committed in uneven pieces)
commit([10, 11]); commit([12, 13, 14]); commit([15]); commit([10, 11, 12])
st = H._REP["r1"]
check("tail holds the last n-1 committed tokens", st["tail"].tolist() == [10, 11, 12])
#            node:  0   1   2   3   4   5   6
tok = torch.tensor([12, 13, 99, 14, 15, 16, 13])
par = torch.tensor([-1, 0, 0, 1, 3, 4, 2])
rep = H._rep_children(st, tok, par)
check("child completing 10 11 12 13 is flagged", bool(rep[1]))
check("novel child 10 11 12 99 is not flagged", not bool(rep[2]))
check("deeper repeat 11 12 13 14 flagged (window crosses the root)", bool(rep[3]))
check("12 13 14 15 flagged (window inside the tree)", bool(rep[4]))
check("13 14 15 16 not flagged (never committed)", not bool(rep[5]))
check("12 99 13 not flagged (branch through the novel token)", not bool(rep[6]))
check("root is never flagged", not bool(rep[0]))

# full sample_nodes: nucleus would relax node 1 (13) and node 2 (99); with the guard node 1 is not relaxed.
n = 7
probs = torch.zeros(1, n, V)
probs[0, 0, 13] = 0.6; probs[0, 0, 99] = 0.3; probs[0, 0, 50] = 0.1      # root row: draw = 50 (set below)
probs[0, 1, 14] = 0.9; probs[0, 1, 51] = 0.1                           # after 13: 14 is a repeat, draw = 51
probs[0, 2, 13] = 0.5; probs[0, 2, 52] = 0.5                           # after 99: 13 is NOT a repeat (12 99 13), draw 52
for j in range(3, n):
    probs[0, j, 60 + j] = 1.0
draw = torch.tensor([[50, 51, 52, 63, 64, 65, 66]])
STATE["probs"] = probs; STATE["draw"] = draw
pred = H.sample_nodes(None, None, bs=1, n_nodes=n, draft_input=None, reqs=reqs, positions=None, q_pass=False,
                      tokens=tok.unsqueeze(0), parents=par.unsqueeze(0))
check("root: 13 would repeat -> not relaxed; next acceptable child 99 is taken", int(pred[0, 0]) == 99)
check("node 1: its only child 14 repeats -> falls back to the lossless draw 51", int(pred[0, 1]) == 51)
check("node 2: child 13 after the novel 99 is relaxed as usual", int(pred[0, 2]) == 13)
check("withheld counter counts the two blocked relaxations", int(H._LAST["rep"]) >= 2)
predq = H.sample_nodes(None, None, bs=1, n_nodes=n, draft_input=None, reqs=reqs, positions=None, q_pass=True,
                       tokens=tok.unsqueeze(0), parents=par.unsqueeze(0))
check("Q pass is not guarded (root takes its first child 13)", int(predq[0, 0]) == 13)
# guard off -> plain nucleus
json.dump({"mode": "nucleus", "norep": 0}, open(ctl.name, "w")); os.utime(ctl.name, ns=(1, 10 ** 18)); H.begin_extend(reqs)
pred0 = H.sample_nodes(None, None, bs=1, n_nodes=n, draft_input=None, reqs=reqs, positions=None, q_pass=False,
                       tokens=tok.unsqueeze(0), parents=par.unsqueeze(0))
check("norep 0: root takes 13, node 1 takes 14", int(pred0[0, 0]) == 13 and int(pred0[0, 1]) == 14)
# a new request starts clean
class Req2:
    rid = "r2"
H.begin_extend([Req2()])
check("state of a finished request is dropped", "r1" not in H._REP)
# hash collisions: 30000 random 12-grams recorded, how many of 30000 fresh ones are flagged
json.dump({"mode": "nucleus", "norep": 12}, open(ctl.name, "w")); os.utime(ctl.name, ns=(1, 2 * 10 ** 18)); H.begin_extend([Req2()])
st2 = H._rep_state("r2", 12, torch.device("cpu"))
g = torch.Generator().manual_seed(1)
win = torch.randint(0, 248000, (30000, 12), generator=g)
st2["bits"].scatter_reduce_(0, H._gram_idx(win), torch.ones(30000, dtype=torch.uint8), reduce="amax")
fresh = torch.randint(0, 248000, (30000, 12), generator=g)
fp = float((st2["bits"][H._gram_idx(fresh)] > 0).float().mean())
check(f"false-positive rate with 30000 recorded 12-grams is small ({fp:.4f})", fp < 0.01)
check("recorded grams are all found", bool((st2["bits"][H._gram_idx(win)] > 0).all()))
# prompt seeding: a request whose prompt ends with ... 30 31 32 33 34 | and earlier contained 31 32 33 34 35
class Req3:
    rid = "r3"
    origin_input_ids = [1, 2, 31, 32, 33, 34, 35, 36, 3, 4, 30, 31, 32, 33, 34]
    output_ids = []
json.dump({"mode": "nucleus", "norep": 4}, open(ctl.name, "w")); os.utime(ctl.name, ns=(1, 3 * 10 ** 18)); H.begin_extend([Req3()])
st3 = H._rep_state("r3", 4, torch.device("cpu"), Req3())
check("tail is the end of the prompt", st3["tail"].tolist() == [32, 33, 34])
tok3 = torch.tensor([34, 35, 77, 36]); par3 = torch.tensor([-1, 0, 0, 1])
rep3 = H._rep_children(st3, tok3, par3)
check("continuing a prompt sequence (32 33 34 35) is flagged", bool(rep3[1]))
check("novel continuation (32 33 34 77) is not", not bool(rep3[2]))
check("33 34 35 36 flagged (prompt sequence, window through the tree)", bool(rep3[3]))
json.dump({"mode": "nucleus", "norep": 4, "norep_prompt": 0}, open(ctl.name, "w")); os.utime(ctl.name, ns=(1, 4 * 10 ** 18))
class Req4(Req3):
    rid = "r4"
H.begin_extend([Req4()])
st4 = H._rep_state("r4", 4, torch.device("cpu"), Req4())
check("norep_prompt 0: prompt sequences are not recorded, tail still set", (not bool(H._rep_children(st4, tok3, par3)[1])) and st4["tail"].tolist() == [32, 33, 34])
print("ALL PASS" if ok_all else "SOME FAILED")
os.unlink(ctl.name)
sys.exit(0 if ok_all else 1)
