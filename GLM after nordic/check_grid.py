# import nibabel as nib
# import numpy as np

# pre = nib.load(r"D:/NORDIC_Subject1_project/NEW Regstration files/warp_outputs/pre_vol0000.nii.gz")
# post = nib.load(r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run1_mag_fixhdr.nii.gz")

# print("PRE shape :", pre.shape)
# print("POST shape:", post.shape)
# print("same_shape:", pre.shape == post.shape)
# print("same_zooms:", np.allclose(pre.header.get_zooms()[:3], post.header.get_zooms()[:3]))
# print("same_affine:", np.allclose(pre.affine, post.affine))
# print("pre q/s:", int(pre.header["qform_code"]), int(pre.header["sform_code"]))
# print("post q/s:", int(post.header["qform_code"]), int(post.header["sform_code"]))

import nibabel as nib
import numpy as np


pre = nib.load(r"D:/NORDIC_Subject1_project/NEW Regstration files/warp_outputs/pre_vol0000.nii.gz")
post = nib.load(r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run1_mag_fixhdr.nii.gz")

post0 = post.slicer[..., 0]

print("PRE shape :", pre.shape)
print("POST0 shape:", post0.shape)
print("same_shape:", pre.shape == post0.shape)
print("same_zooms:", np.allclose(pre.header.get_zooms()[:3], post0.header.get_zooms()[:3]))
print("same_affine:", np.allclose(pre.affine, post0.affine))
print("pre q/s:", int(pre.header["qform_code"]), int(pre.header["sform_code"]))
print("post0 q/s:", int(post0.header["qform_code"]), int(post0.header["sform_code"]))