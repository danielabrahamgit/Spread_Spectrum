import sys
import os
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath('../'))


import numpy as np
from utils.sig_utils import sig_utils
from utils.SDR_utils import RTL_utils, UHD_utils
from utils.SSM_decoder import SSM_decoder

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# --------------- CHANGE SEQUENCE PARAMTERS HERE ---------------
center_freq = 134.5e6
rx_rate = 1e6
rx_gain = 10
prnd_seq_len = 2**14
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
read_time = 3
num_samples = read_time * rx_rate
# --------------------------------------------------------------

rtl = RTL_utils()
sig = rtl.rtl_read(
	freq=center_freq, 
	rate=rx_rate, 
	gain=rx_gain,
	num_samples=num_samples)

sig -= np.mean(sig)


# rtl.view_spectrum(sig, center_freq, rx_rate)

M = 1
# sig = sig_utils.my_resample(sig, 1, M)
# rx_rate = rx_rate // M

prnd_seq = sig_utils.prnd_gen(seq_len=prnd_seq_len, type=prnd_type, mode=prnd_mode, seed=prnd_seed)

dec= SSM_decoder(rx_rate, prnd_seq, pt_bw=1e6)

# chop = prnd_seq_len // (10 * M)
chop = 512
# # print(chop, chop * M)
est = dec.motion_estimate_iq(sig, chop=chop, mode='RSSM', normalize=False)
t = np.arange(len(est)) * chop / rx_rate

plt.plot(t, est)
plt.show()