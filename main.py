import numpy as np
import matplotlib.pyplot as plt
from utils import sig_utils, MR_utils
from PIL import Image

# Uncertanty in TR
tr_uncert = 0.01

# Load MR image
im = np.array(Image.open('images/brain.png'))
# ksp = np.load('kspace/500_30sec.npz')['ksp'][:,:,0,3].T
ksp = np.random.normal(0, 2, im.shape) + 1j * np.random.normal(0, 2, im.shape)

# Initialize MR object with the parameters below
mr = MR_utils(tr=34.876e-3, bwpp=250e3/256, fc=127.8e6)

# Motion modulation signal
def mod(t):
	return 1000 + 100 * np.sin(2 * np.pi * t / (50 * mr.TR))

# Load an image and add a pilot tone to it. 
mr.load_image(im)
#mr.load_kspace(ksp)
a, b = mr.add_PT(127.8e6 - 100e3, tr_uncert=tr_uncert, modulation=mod)
motion = mr.motion_extract()
plt.plot(np.abs(motion))
mr.MRshow(drng=1e-6, log=True)
