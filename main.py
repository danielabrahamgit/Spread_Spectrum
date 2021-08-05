import numpy as np
import matplotlib.pyplot as plt
from utils import MR_utils
from PIL import Image

# Uncertanty in TR
tr_uncert = 0
# Pilot tone frequency
fpt = 0

# Load MR image
# im = np.array(Image.open('images/brain.png'))
im = np.load('images/brain.npz')['im']
# ksp = np.random.normal(0, 2, im.shape) + 1j * np.random.normal(0, 2, im.shape)

# Initialize MR object with the parameters below
mr = MR_utils(tr=34.876e-3, bwpp=250e3/max(im.shape), fc=127.8e6)

# Load Data into MR object
mr.load_image(im)
# mr.load_kspace(ksp)

# Motion modulation signal
def mod(t):
	return 1e1 #* (2 + np.sin(2 * np.pi * t / (50 * mr.TR)))
# mod = 20 + 10 * np.sin(2 * np.pi * np.arange(mr.ksp.shape[0]) / (50 * mr.TR * mr.fs))
# Spread spectrum modulation PRN sequence
mr.prnd_seq_gen(seq_len=10000)

# Add Pilot tone (with modulation) and extract motion + image
ksp_std = mr.get_ksp_std()
a, b = mr.add_PT(fpt, tr_uncert=tr_uncert, modulation=mod)


plt.plot(np.abs(np.mean(mr.prnd_mat * np.real(mr.ksp), axis=1)))
plt.plot(mr.true_motion)
plt.figure()


# Plot motion estimates
motion = mr.motion_extract(fpt=fpt)
true_motion = mr.true_motion
plt.subplot(2,1,1)
plt.title('Standard Deviation of Each Readout')
plt.xlabel('Phase Encode #')
plt.ylabel('$\sigma$')
plt.plot(ksp_std)
plt.subplot(2,1,2)
plt.title('Inner Product Estimate')
plt.xlabel('Phase Encode #')
plt.plot(np.abs(motion), label='Inner Product Estimate')
plt.plot(np.abs(true_motion), label='True Modulation')
plt.legend()

# Show eveything
mr.MRshow(drng=1e-6, log_ksp=True, log_ro=True, log_im=False)
