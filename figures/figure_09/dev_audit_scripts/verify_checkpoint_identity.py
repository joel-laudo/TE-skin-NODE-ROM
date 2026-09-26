"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
one-off tensor-equality check that each reported `*_BEST.pt` checkpoint is
bit-identical to the corresponding `epoch_NNN.pt` snapshot saved during
training, for the 360-epoch and 560-epoch delayed-decay runs of Models A
and D. Confirms the cached rollout evaluations used elsewhere really
correspond to the epoch they claim to. Not part of reproducing any
reported figure. Depends on the private project's `epoch_checkpoints/`
archives, which are not included in this repo."""
import torch

pairs = [
    ("A_360_BEST_vs_epoch356",
     "FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r9_BEST.pt",
     "FINAL_models/Model_A_ablation_fullrollout_v2/epoch_checkpoints/epoch_356.pt"),
    ("D_360_BEST_vs_epoch357",
     "FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r9_BEST.pt",
     "FINAL_models/Model_D_ablation_fullrollout_v2/epoch_checkpoints/epoch_357.pt"),
    ("A_delayed_BEST_vs_epoch560",
     "FINAL_models/Model_A_ablation_fullrollout_v2_delayeddecay560/Model_A_Vanilla_r9_BEST.pt",
     "FINAL_models/Model_A_ablation_fullrollout_v2_delayeddecay560/epoch_checkpoints/epoch_560.pt"),
    ("D_delayed_BEST_vs_epoch558",
     "FINAL_models/Model_D_ablation_fullrollout_v2_delayeddecay560/Model_D_CNN_r9_BEST.pt",
     "FINAL_models/Model_D_ablation_fullrollout_v2_delayeddecay560/epoch_checkpoints/epoch_558.pt"),
]

for name, p_best, p_epoch in pairs:
    ck_best = torch.load(p_best, map_location="cpu")
    ck_epoch = torch.load(p_epoch, map_location="cpu")
    sd_best = ck_best["model_state_dict"]
    sd_epoch = ck_epoch["model_state_dict"]
    same_keys = set(sd_best.keys()) == set(sd_epoch.keys())
    all_equal = same_keys and all(torch.equal(sd_best[k], sd_epoch[k]) for k in sd_best)
    print(f"{name}: same_keys={same_keys} tensor_identical={all_equal}")
