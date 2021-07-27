import numpy as np
import matplotlib.pyplot as plt
from utils import MR_utils
from PIL import Image

# Uncertanty in TR
tr_uncert = 0.00
# Pilot tone frequency
fpt = 0

# Initialize MR object with the parameters below
mr = MR_utils(tr=34.876e-3, bwpp=250e3/256, fc=127.8e6)

# Load MR image
im = np.array(Image.open('images/brain.png'))
# ksp = np.load('kspace/500_30sec.npz')['ksp'][:,:,0,3].T
ksp = np.random.normal(0, 2, im.shape) + 1j * np.random.normal(0, 2, im.shape)

# Load Data into MR object
mr.load_image(im)
# mr.load_kspace(ksp)

# Motion modulation signal
def mod(t):
	return 1e3 * (10 +  np.sin(2 * np.pi * t / (50 * mr.TR)))

# Spread spectrum modulation PRN sequence
mr.prnd_seq_gen(p=0.5)
# prnd_seq = mr.prnd_seq_gen(start_state=0xACE9)
# mag = np.random.uniform(1e-1, 1, len(prnd_seq))
# mag = np.random.uniform(1, 3, len(prnd_seq))
# prnd_seq *= mag
# prnd_seq = None

# Add Pilot tone (with modulation) and extract motion + image
a, b = mr.add_PT(fpt, tr_uncert=tr_uncert, modulation=mod)
motion = mr.motion_extract(fpt=fpt)
plt.title('Extracted Motion Signal')
plt.plot(motion)
mr.MRshow(drng=1e-6, log_ksp=True, log_ro=True, log_im=False)
