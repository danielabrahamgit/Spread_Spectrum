import numpy as np
import matplotlib.pyplot as plt
from utils import MR_utils
from PIL import Image

# Uncertanty in TR
tr_uncert = 0.0
# Pilot tone frequency
fpt = 0

# Load MR image
# im = np.array(Image.open('images/brain.png'))
# zer = np.zeros((im.shape[0]+1, im.shape[1]+1), dtype=im.dtype)
# zer[:im.shape[0],:im.shape[1]] = im
# im = zer
im = np.load('images/brain.npz')['im']
# ksp = np.random.normal(0, 2, im.shape) + 1j * np.random.normal(0, 2, im.shape)

# Initialize MR object with the parameters below
mr = MR_utils(tr=34.876e-3, bwpp=250e3/max(im.shape), fc=127.8e6)

# Load Data into MR object
mr.load_image(im)
# mr.load_kspace(ksp)

# Motion modulation signal
def mod(t):
	return 1e1 * (2 + np.sin(2 * np.pi * t / (50 * mr.TR)))

# Spread spectrum modulation PRN sequence
mr.prnd_seq_gen()

# # Play with correlation
# ksp = np.real(mr.ksp)
# cor = np.abs(mr.prnd_seq @ ksp.T)
# l2 = np.mean(cor ** 2, axis=1)
# l1 = np.mean(cor, axis=1)
# mx = np.max(cor, axis=1)

# lin_comb = l1 + l2 + mx
# min_ind, min_val = np.argmin(lin_comb), np.min(lin_comb)
# print(min_ind, min_val)
# plt.figure()
# with open('scratch.txt', 'w') as f:
# 	f.write('np.array([')	
# 	for v in mr.prnd_seq[min_ind, :][:-1]:
# 		f.write(str(v) + ', ')
# 	f.write(str(mr.prnd_seq[min_ind, -1]) + '])')

# plt.plot(np.abs(np.fft.fftshift(np.fft.fft(mr.prnd_seq[min_ind, :]))))
# plt.figure()
# plt.imshow(np.log(cor))
# plt.show()
# quit()

# Add Pilot tone (with modulation) and extract motion + image
ksp_std = mr.get_ksp_std()
a, b = mr.add_PT(fpt, tr_uncert=tr_uncert, modulation=mod)

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

# Plot corelation of prnd with kspace
plt.figure()
corr = np.abs(mr.prnd_seq @ mr.ksp.T)
plt.imshow(corr)

# Show eveything
mr.MRshow(drng=1e-6, log_ksp=True, log_ro=True, log_im=False)
