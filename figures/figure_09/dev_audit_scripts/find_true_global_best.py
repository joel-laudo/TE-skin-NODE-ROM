"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
combine training-time disp_rmse_eval_log.txt entries with the retrospective
sparse-eval sweep from audit_sparse_eval_delayeddecay.py (epochs 301-500,
previously unevaluated during training) to find the TRUE global minimum
median displacement RMSE across the ENTIRE saved-checkpoint trajectory
(41-560) for each delayed-decay run.

NOTE: audit_sparse_eval_delayeddecay.py's own CSV-writing step crashed after
completing the full sweep (a tuple-unpacking bug: `for ep, val, expected_val in
training_time_vals.items()` -- a 2-tuple unpacked into 3 names). No compute was
lost -- all 160 retrospective results per model are parsed directly from the
run's own stdout logs instead.

Not part of reproducing any reported figure. Read-only: no training, no
checkpoint modification. Depends on the excluded checkpoint archive and
stdout log files not present in this repo.
"""
import re

MODELS = {
    "A": dict(
        disp_log="FINAL_models/Model_A_ablation_fullrollout_v2_delayeddecay560/disp_rmse_eval_log.txt",
        stdout_log="revised_figure9_v2/sparse_eval_A_stdout.log",
        current_best_epoch=560,
        current_best_rmse=0.136109,
    ),
    "D": dict(
        disp_log="FINAL_models/Model_D_ablation_fullrollout_v2_delayeddecay560/disp_rmse_eval_log.txt",
        stdout_log="revised_figure9_v2/sparse_eval_D_stdout.log",
        current_best_epoch=558,
        current_best_rmse=0.134388,
    ),
}


def parse_training_time_log(path):
    pat = re.compile(r"^epoch=(\d+)\s+median_disp_rmse=([\d.eE+\-]+)\s*$")
    out = {}
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def parse_retrospective_stdout(path):
    """Parse lines like:
    '  [1/160] epoch=301 median_disp_rmse=0.7921536226792351 (26.0s, ...)'
    """
    pat = re.compile(r"epoch=(\d+)\s+median_disp_rmse=([\d.eE+\-]+)")
    out = {}
    with open(path) as f:
        for line in f:
            if "median_disp_rmse=None" in line:
                m2 = re.match(r"\s*\[\d+/\d+\]\s*epoch=(\d+)", line)
                if m2:
                    out[int(m2.group(1))] = None
                continue
            m = pat.search(line)
            if m and "[" in line:  # only the "[i/160] epoch=..." progress lines, not the summary header
                out[int(m.group(1))] = float(m.group(2))
    return out


all_results = {}
for fam, cfg in MODELS.items():
    training_vals = parse_training_time_log(cfg["disp_log"])
    retro_raw = parse_retrospective_stdout(cfg["stdout_log"])
    failed = [ep for ep, v in retro_raw.items() if v is None]
    retro_vals = {ep: v for ep, v in retro_raw.items() if v is not None}

    overlap = set(training_vals) & set(retro_vals)
    assert not overlap, f"{fam}: unexpected overlap between training-time and retrospective epochs: {sorted(overlap)}"

    all_vals = dict(training_vals)
    all_vals.update(retro_vals)

    global_best_epoch = min(all_vals, key=all_vals.get)
    global_best_rmse = all_vals[global_best_epoch]

    best_of_originally_evaluated = min(training_vals, key=training_vals.get)
    best_of_originally_evaluated_rmse = training_vals[best_of_originally_evaluated]

    best_of_retro = min(retro_vals, key=retro_vals.get) if retro_vals else None
    best_of_retro_rmse = retro_vals[best_of_retro] if best_of_retro else None

    changed = (global_best_epoch != cfg["current_best_epoch"])

    print(f"=== Model {fam} ===")
    print(f"  n_training_time_evaluated = {len(training_vals)} (expected 152: 52 in 41-300 + 40 in 301-500 + 60 in 501-560)")
    print(f"  n_retrospective_evaluated = {len(retro_vals)} (expected 160), n_failed = {len(failed)}")
    if failed:
        print(f"  FAILED epochs: {failed}")
    print(f"  Total unique epochs covered = {len(all_vals)} (expected 312 = 152 + 160)")
    print(f"  Currently reported BEST: epoch={cfg['current_best_epoch']} rmse={cfg['current_best_rmse']}")
    print(f"  Best among ORIGINALLY evaluated epochs: epoch={best_of_originally_evaluated} rmse={best_of_originally_evaluated_rmse:.6f}")
    print(f"  Best among PREVIOUSLY UNEVALUATED (retrospective) epochs: epoch={best_of_retro} rmse={best_of_retro_rmse:.6f}")
    print(f"  TRUE GLOBAL MINIMUM across full 41-560 trajectory: epoch={global_best_epoch} rmse={global_best_rmse:.6f}")
    print(f"  BEST CHANGED: {changed}")
    diff = best_of_retro_rmse - cfg["current_best_rmse"]
    pct = 100 * diff / cfg["current_best_rmse"]
    print(f"  Best retrospective vs. current BEST: diff={diff:+.6f} ({pct:+.2f}%) "
          f"{'-- retrospective BEATS current BEST' if diff < 0 else '-- retrospective does NOT beat current BEST'}")
    print()

    all_results[fam] = dict(
        training_vals=training_vals, retro_vals=retro_vals, all_vals=all_vals,
        global_best_epoch=global_best_epoch, global_best_rmse=global_best_rmse,
        changed=changed, failed=failed,
    )

import json
with open("revised_figure9_v2/true_global_best_results.json", "w") as f:
    json.dump({
        fam: dict(
            global_best_epoch=r["global_best_epoch"], global_best_rmse=r["global_best_rmse"],
            changed=r["changed"], failed=r["failed"],
            n_training=len(r["training_vals"]), n_retro=len(r["retro_vals"]),
        ) for fam, r in all_results.items()
    }, f, indent=2)
print("saved revised_figure9_v2/true_global_best_results.json")
