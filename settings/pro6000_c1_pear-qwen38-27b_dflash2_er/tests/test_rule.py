"""CPU test of scripts/rule.py (stdlib only): the "er" arm is the measured thr0.3+nohes+norep+retry control config, and
thresholds below 0.3 are refused. Run: python3 settings/pro6000_c1_pear-qwen38-27b_dflash2_er/tests/test_rule.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import rule  # noqa: E402

er = rule.arm_cfg("er")
assert er == rule.arm_cfg("thr0.3+nohes+norep+retry")
assert er["mode"] == "thr" and er["theta"] == 0.3, er
assert er["suppress"] == rule.HES_IDS and er["suppress_max"] == 0.999, er
assert er["norep"] == 8, er
assert er["strict"] == list(range(248044, 248320)), "structural-token guard"
assert rule.wants_retry("er") and not rule.wants_retry("lossless")
assert rule.arm_cfg("lossless") == {"mode": "off"}
for bad in ("thr0.1", "thr0.29+nohes", "rel0.2", "nucleus"):
    try:
        rule.arm_cfg(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError(f"{bad} should be refused without --allow-below-0.3")
assert rule.arm_cfg("thr0.1", allow_low=True)["theta"] == 0.1
print("ALL PASS")
