import sys
import os
from matplotlib import pyplot as plt
from math import ceil

sys.path.append(os.path.abspath('../'))


import numpy as np
from scipy import signal
from numpy.fft import fft, fftshift
from utils.sig_utils import sig_utils
from utils.SDR_utils import RTL_utils, UHD_utils
from utils.SSM_decoder import SSM_decoder

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# --------------- CHANGE SEQUENCE PARAMTERS HERE ---------------
center_freq = 150e6
rx_rate = 1e6
rx_gain = 0
prnd_seq_len = 256
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
num_samples = prnd_seq_len * 100
M = 1
# --------------------------------------------------------------

# Copy prnd seq
prnd_seq = sig_utils.prnd_gen(
			seq_len=prnd_seq_len, 
			type=prnd_type, 
			mode=prnd_mode, 
			seed=prnd_seed
)
# prnd_seq = np.repeat(prnd_seq, 2)


# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)
iq_sig = uhd.uhd_read(
			freq=center_freq,
			rate=rx_rate * M,
			gain=rx_gain,
			duration=num_samples / (rx_rate * M),
			arg='3215B78',
			use_sdr=True
)
iq_sig = signal.resample_poly(iq_sig, 1, M)

dec = SSM_decoder(
			mr_bw=rx_rate, 
			prnd_seq=prnd_seq, 
			pt_fc=0, 
			pt_bw=rx_rate,
			doppler_range=5e3
)


# est = dec.motion_estimate_iq(
# 			iq_sig, 
# 			mode='RSSM', 
# 			normalize=False, 
# 			chop=prnd_seq_len
# )


sig_utils.view_spectrum(
			iq_sig=iq_sig,
			freq=center_freq,
			rate=rx_rate,
			n_avg=5,
			n_fft=1024,
			eps=1e-6,
			log=False
)

# N = 10
# iq_sig = np.mean(np.reshape(iq_sig[:(len(iq_sig) // N) * N], (N, -1)), axis=0)

# iq_sig *= np.exp(1j * np.arange(len(iq_sig)) * 2 * np.pi * (-5e3) / rx_rate)

# cor = sig_utils.my_cor(iq_sig, prnd_seq)

# plt.subplot(3,1,1)
# plt.plot(cor.real)
# plt.subplot(3,1,2)
# plt.plot(cor.imag)
# plt.subplot(3,1,3)
# plt.plot(np.abs(cor))
# plt.show()


est = dec.motion_estimate_iq(
				iq_sig=iq_sig, 
				chop=prnd_seq_len, 
				normalize=False,
				mode='RSSM'
)
plt.plot(est)
plt.show()
