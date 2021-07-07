import nrrd
import numpy as np
import matplotlib.pyplot as plt
from utils import sig_utils, MR_utils
from PIL import Image


im = np.array(Image.open('images/brain.png'))

mr = MR_utils(tr=34e-3, bwpp=250e3/256, fc=127.8e6)
mr.load_image(im)
mr.add_PT(127.6e6)
mr.MRshow()
quit()

F_TONE = 300
sig_utils.Fs = 1e3

# Set random seed so that we get same results everytime
np.random.seed(12)

# Get pure tone signal with noise
t, sig = sig_utils.gen_tone(F_TONE, -2, 2)
# sig += sig_utils.gen_guassian_noise(len(sig))

# Signal length
n_sig = len(sig)

# Generates sequence of 1 and -1
flips = np.random.binomial(1, 0.01, n_sig)
prnd = []
negative = True
for flip in flips:
	if flip:
		negative = not negative
	if negative:
		prnd.append(-1)
	else:
		prnd.append(1)
prnd = np.array(prnd)

# Plot resulting spectrum
sig_utils.plot_fft(sig * prnd)