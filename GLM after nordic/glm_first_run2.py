 #!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel

# =========================
# EDIT THESE ONLY
# =========================
FUNC = r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run2_mag_fixhdr.nii.gz"
OUT_DIR = r"D:/NORDIC_Subject1_project/GLM after nordic/glm_postnordic_run2"
TR = 3.0
MASK = None
CONFOUNDS = None
# =========================
MASK = None
CONFOUNDS = None
EVENTS = pd.DataFrame([
    [0.025552034, 3.000006676, "anticipation_low_cognition"],
    [3.02555871, 24.00002027, "pain_low_cogn_low_pain"],
    [27.02557898, 2.958322048, "rating"],
    [57.03393936, 3.000001907, "anticipation_high_cognition"],
    [60.03394127, 24.00002313, "pain_high_cogn_high_pain"],
    [84.0339644, 5.241750717, "rating"],
    [123.0340059, 2.999999285, "anticipation_high_cognition"],
    [126.0340052, 24.00002694, "pain_high_cogn_low_pain"],
    [150.0340321, 4.750077009, "rating"],
    [189.0340669, 3.0, "anticipation_high_cognition"],
    [192.0340669, 24.00002527, "pain_high_cogn_low_pain"],
    [216.0340922, 3.42500782, "rating"],
    [249.0341272, 3.000001192, "anticipation_high_cognition"],
    [252.0341284, 24.00002241, "pain_high_cogn_high_pain"],
    [276.0341508, 3.575156212, "rating"],
    [312.0341814, 3.000003576, "anticipation_low_cognition"],
    [315.0341849, 24.00002432, "pain_low_cogn_high_pain"],
    [339.0342093, 4.025078773, "rating"],
    [372.0342402, 3.000001907, "anticipation_high_cognition"],
    [375.0342422, 24.00002456, "pain_high_cogn_low_pain"],
    [399.0342667, 5.308424234, "rating"],
    [429.034296, 3.000001431, "anticipation_low_cognition"],
    [432.0342975, 24.00002289, "pain_low_cogn_high_pain"],
    [456.0343204, 4.375053644, "rating"],
    [486.0343482, 3.000003099, "anticipation_low_cognition"],
    [489.0343513, 24.00002337, "pain_low_cogn_low_pain"],
    [513.0343747, 7.466667414, "rating"],
    [549.03441, 3.000002861, "anticipation_high_cognition"],
    [552.0344129, 24.00002313, "pain_high_cogn_high_pain"],
    [576.034436, 3.766850948, "rating"],
    [609.034467, 3.000002146, "anticipation_low_cognition"],
    [612.0344691, 24.00002408, "pain_low_cogn_high_pain"],
    [636.0344932, 4.400205851, "rating"],
    [672.0345297, 3.000002861, "anticipation_low_cognition"],
    [675.0345325, 24.0000217, "pain_low_cogn_low_pain"],
    [699.0345542, 5.041758776, "rating"],
    [729.0345802, 3.000008821, "anticipation_low_cognition"],
    [732.0345891, 24.00002027, "pain_low_cogn_low_pain"],
    [756.0346093, 6.466927767, "rating"],
    [789.0346401, 3.000001431, "anticipation_high_cognition"],
    [792.0346415, 24.00002766, "pain_high_cogn_high_pain"],
    [816.0346692, 5.258458853, "rating"],
    [846.0346975, 3.000001669, "anticipation_high_cognition"],
    [849.0346992, 24.00002575, "pain_high_cogn_low_pain"],
    [873.034725, 4.116722107, "rating"],
    [909.034754, 3.000004292, "anticipation_low_cognition"],
    [912.0347583, 24.00002122, "pain_low_cogn_low_pain"],
    [936.0347795, 5.016639471, "rating"],
    [969.0348103, 3.000003338, "anticipation_low_cognition"],
    [972.0348136, 24.00002193, "pain_low_cogn_high_pain"],
    [996.0348356, 4.516943455, "rating"],
    [1035.034874, 3.000003338, "anticipation_low_cognition"],
    [1038.034878, 24.00002146, "pain_low_cogn_high_pain"],
    [1062.034899, 4.191714764, "rating"],
    [1101.034936, 3.000003815, "anticipation_high_cognition"],
    [1104.03494, 24.00002265, "pain_high_cogn_low_pain"],
    [1128.034962, 3.758345604, "rating"],
    [1167.034999, 3.000004292, "anticipation_high_cognition"],
    [1170.035004, 24.00002265, "pain_high_cogn_high_pain"],
    [1194.035026, 4.125161648, "rating"],
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