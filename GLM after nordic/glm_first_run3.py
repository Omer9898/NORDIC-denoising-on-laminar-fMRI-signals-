    #!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel

# =========================
# EDIT THESE ONLY
# =========================
FUNC = r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run3_mag_fixhdr.nii.gz"
OUT_DIR = r"D:/NORDIC_Subject1_project/GLM after nordic/glm_postnordic_run3"
TR = 3.0
MASK = None
CONFOUNDS = None
# =========================
MASK = None
CONFOUNDS = None
EVENTS = pd.DataFrame([
    [0.023189783, 3.000006437, "anticipation_high_cognition"],
    [3.02319622, 24.00001645, "pain_high_cogn_high_pain"],
    [27.02321267, 4.533311367, "rating"],
    [63.03157973, 3.000004768, "anticipation_high_cognition"],
    [66.0315845, 24.00002122, "pain_high_cogn_high_pain"],
    [90.03160572, 4.033244371, "rating"],
    [120.0316372, 2.999999762, "anticipation_low_cognition"],
    [123.031637, 24.00002289, "pain_low_cogn_low_pain"],
    [147.0316598, 4.49184823, "rating"],
    [177.0316923, 3.000002861, "anticipation_high_cognition"],
    [180.0316951, 24.00002217, "pain_high_cogn_high_pain"],
    [204.0317173, 5.833322763, "rating"],
    [237.0317519, 2.999997377, "anticipation_low_cognition"],
    [240.0317492, 24.00002337, "pain_low_cogn_high_pain"],
    [264.0317726, 3.316745281, "rating"],
    [294.0318022, 3.000003338, "anticipation_high_cognition"],
    [297.0318055, 24.00002193, "pain_high_cogn_low_pain"],
    [321.0318274, 7.641726494, "rating"],
    [360.0318642, 3.000002861, "anticipation_low_cognition"],
    [363.031867, 24.00002313, "pain_low_cogn_low_pain"],
    [387.0318902, 4.333276749, "rating"],
    [420.0319259, 2.999999046, "anticipation_high_cognition"],
    [423.031925, 24.00002217, "pain_high_cogn_high_pain"],
    [447.0319471, 3.733274937, "rating"],
    [480.0319793, 3.000001669, "anticipation_low_cognition"],
    [483.031981, 24.0000267, "pain_low_cogn_high_pain"],
    [507.0320077, 4.875305414, "rating"],
    [546.0320423, 3.000001431, "anticipation_high_cognition"],
    [549.0320437, 24.00002933, "pain_high_cogn_low_pain"],
    [573.032073, 4.71665287, "rating"],
    [609.0321064, 3.000002861, "anticipation_low_cognition"],
    [612.0321093, 24.00001907, "pain_low_cogn_low_pain"],
    [636.0321283, 5.908675671, "rating"],
    [672.0321624, 3.000006437, "anticipation_low_cognition"],
    [675.0321689, 24.00002337, "pain_low_cogn_high_pain"],
    [699.0321922, 5.225141764, "rating"],
    [738.0322247, 3.000003338, "anticipation_low_cognition"],
    [741.032228, 24.00002313, "pain_low_cogn_high_pain"],
    [765.0322511, 4.116830826, "rating"],
    [795.03228, 3.000003099, "anticipation_high_cognition"],
    [798.0322831, 24.00002551, "pain_high_cogn_low_pain"],
    [822.0323086, 2.550140381, "rating"],
    [855.0323391, 3.000000715, "anticipation_high_cognition"],
    [858.0323398, 24.00002337, "pain_high_cogn_low_pain"],
    [882.0323632, 5.933324814, "rating"],
    [915.0323961, 3.000004292, "anticipation_high_cognition"],
    [918.0324004, 24.00002074, "pain_high_cogn_high_pain"],
    [942.0324211, 4.100085974, "rating"],
    [972.0324516, 3.000000954, "anticipation_low_cognition"],
    [975.0324526, 24.00002408, "pain_low_cogn_low_pain"],
    [999.0324767, 2.883629322, "rating"],
    [1038.032518, 3.000001907, "anticipation_low_cognition"],
    [1041.03252, 24.0000186, "pain_low_cogn_high_pain"],
    [1065.032538, 3.858484507, "rating"],
    [1101.032574, 3.000002384, "anticipation_high_cognition"],
    [1104.032577, 24.00002456, "pain_high_cogn_low_pain"],
    [1128.032601, 3.241569757, "rating"],
    [1164.032637, 2.999999762, "anticipation_low_cognition"],
    [1167.032637, 24.00002193, "pain_low_cogn_low_pain"],
    [1191.032659, 7.050327301, "rating"],
], columns=["onset", "duration", "trial_type"])



DEFAULT_CONTRASTS = {
    "maineffectofcognition":
        "pain_high_cogn_high_pain + pain_high_cogn_low_pain - pain_low_cogn_high_pain - pain_low_cogn_low_pain",
    "maineffectofpain":
        "pain_high_cogn_high_pain - pain_high_cogn_low_pain + pain_low_cogn_high_pain - pain_low_cogn_low_pain",
    "interactioneffect":
        "- pain_high_cogn_high_pain + pain_high_cogn_low_pain + pain_low_cogn_high_pain - pain_low_cogn_low_pain",
}

def tsnr(func_path, mask_path=None):
    img = nib.load(func_path)
    d = img.get_fdata(dtype=np.float32)
    mean = d.mean(-1)
    sd = d.std(-1)
    t = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    if mask_path:
        m = nib.load(mask_path).get_fdata() > 0
        return float(np.median(t[m]))
    return float(np.median(t[mean > mean.mean()]))

def main():
    if not os.path.exists(FUNC):
        raise FileNotFoundError(f"File not found: {FUNC}")

    os.makedirs(OUT_DIR, exist_ok=True)

    img = nib.load(FUNC)
    if img.ndim != 4:
        raise ValueError(f"FUNC must be 4D, got {img.shape}")
    n_scans = img.shape[3]

    if len(EVENTS) == 0:
        raise ValueError("EVENTS is empty. Paste the correct table for this run.")

    needed = {"onset", "duration", "trial_type"}
    if not needed.issubset(EVENTS.columns):
        raise ValueError("EVENTS must have columns: onset, duration, trial_type")

    EVENTS2 = EVENTS[["onset", "duration", "trial_type"]].copy()
    EVENTS2["trial_type"] = EVENTS2["trial_type"].astype(str)

    if CONFOUNDS:
        confounds = pd.read_csv(CONFOUNDS, sep=None, engine="python")
        if len(confounds) != n_scans:
            raise ValueError(f"confounds rows ({len(confounds)}) != n_scans ({n_scans})")
    else:
        confounds = None

    glm = FirstLevelModel(
        t_r=TR,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=1.0 / 180.0,
        noise_model="ar1",
        smoothing_fwhm=None,
        mask_img=MASK,
        minimize_memory=False,
        signal_scaling=0,
    )

    print(f"[glm] file: {FUNC}")
    print(f"[glm] fitting: n_scans={n_scans} TR={TR}s conditions={sorted(EVENTS2['trial_type'].unique())}")
    glm.fit(FUNC, events=EVENTS2, confounds=confounds)

    for name, expr in DEFAULT_CONTRASTS.items():
        eff = glm.compute_contrast(expr, output_type="effect_size")
        z = glm.compute_contrast(expr, output_type="z_score")
        stat = glm.compute_contrast(expr, output_type="stat")

        safe = name.replace(" ", "").replace("-", "_minus_")
        nib.save(eff, os.path.join(OUT_DIR, f"{safe}_effect.nii.gz"))
        nib.save(z, os.path.join(OUT_DIR, f"{safe}_z.nii.gz"))
        nib.save(stat, os.path.join(OUT_DIR, f"{safe}_t.nii.gz"))
        print(f"[glm] {name} saved")

    EVENTS2.to_csv(os.path.join(OUT_DIR, "events_used.csv"), index=False)
    glm.design_matrices_[0].to_csv(os.path.join(OUT_DIR, "design_matrix.csv"), index=False)

    try:
        print(f"[qc] native tSNR (median) = {tsnr(FUNC, MASK):.2f}")
    except Exception as e:
        print(f"[qc] tSNR skipped: {e}")

    print(f"\n[done] outputs -> {OUT_DIR}")

if __name__ == "__main__":
    main()