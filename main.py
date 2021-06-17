import numpy as np
import matplotlib.pyplot as plt
from sig_utils import sig_utils

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