import sys
import os
sys.path.append(os.path.abspath('../'))


import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift, fft
from scipy import signal
from utils.sig_utils import sig_utils
from utils.SDR_utils import UHD_utils
from utils.SSM_decoder import SSM_decoder


# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# --------------- CHANGE SEQUENCE PARAMTERS HERE ---------------
seq_id = '/scans/cory_sine_5hz'
center_freq = 134.5e6
tx_rate = 1e6
tx_gain = 30
prnd_seq_len = 1024
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
L = 1
# --------------------------------------------------------------

# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)

# Store all parameters in dictionary (Eventually text file)
seq_data = {}
seq_data['prnd_seq_len'] = prnd_seq_len
seq_data['prnd_type'] = prnd_type
seq_data['prnd_mode'] = prnd_mode
seq_data['prnd_seed'] = prnd_seed
seq_data['upsample_factor'] = L
seq_data['center_freq'] = center_freq
seq_data['tx_rate'] = tx_rate
seq_data['tx_gain'] = tx_gain

# SSM iq_sig gen
n = np.arange(prnd_seq_len)
prnd_seq = sig_utils.prnd_gen(seq_len=prnd_seq_len, type=prnd_type, mode=prnd_mode, seed=prnd_seed)
# iq_sig = np.exp(np.arange(prnd_seq_len) * 2j * np.pi * 0 / tx_rate)
# iq_sig = np.linspace(-0.5, 0.5, prnd_seq_len) ** 2 
# iq_sig = signal.firwin(prnd_seq_len, 0.1, fs=2).astype(np.complex128)
# iq_sig = iq_sig ** 2 / 0.1

Npts = 100
iq_sig = np.tile(prnd_seq, 100)
A = np.repeat(np.arange(Npts) + 1, prnd_seq_len)
iq_sig *= A

# iq_sig *= np.exp(1j * n * 2 * np.pi * (200e3) / tx_rate)
iq_sig = iq_sig.astype(np.complex128)


# dec= SSM_decoder(tx_rate, prnd_seq, pt_bw=1e6)

# # chop = prnd_seq_len // (10 * M)
# chop = prnd_seq_len
# # # print(chop, chop * M)
# est = dec.motion_estimate_iq(iq_sig, chop=chop, mode='RSSM', normalize=False)
# t = np.arange(len(est)) * chop / tx_rate

# plt.plot(t, est)
# plt.show()
# quit()

f = np.linspace(-tx_rate/2e3, tx_rate/2e3, len(iq_sig))
plt.ylabel('Magnitude')
plt.xlabel('Frequency (kHz)')
plt.plot(f, np.abs(fftshift(fft(iq_sig))))
plt.show()


# Resample If needed
if L != 1:
	iq_sig_new_rate = np.zeros(len(iq_sig) * L)
	iq_sig_new_rate[::L] = iq_sig
	h_lpf = signal.firwin(129, tx_rate/2, fs=tx_rate * L)
	iq_sig = np.convolve(iq_sig_new_rate, h_lpf, mode='same')
	tx_rate *= L

# Save to text file
s = ''
for key in seq_data.keys():
	s += key + ': '
	s += str(seq_data[key])
	s += '\n\n'
with open(uhd.PY_DIR +  seq_id + '.txt', 'w') as f:
	f.write(s)

# Transmit iq signal infinitely
uhd.uhd_write(
	iq_sig=iq_sig,
	freq=center_freq,
	rate=tx_rate,
	gain=tx_gain,
	repeat=True)
