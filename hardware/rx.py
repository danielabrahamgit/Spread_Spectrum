import sys
import os
from matplotlib import pyplot as plt
from math import ceil

sys.path.append(os.path.abspath('../'))


import numpy as np
from numpy.fft import fft, fftshift
from utils.sig_utils import sig_utils
from utils.SDR_utils import RTL_utils, UHD_utils
from utils.SSM_decoder import SSM_decoder

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# --------------- CHANGE SEQUENCE PARAMTERS HERE ---------------
center_freq = 152.5e6
rx_rate = 1e6
rx_gain = 5
prnd_seq_len = 1024
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
read_time = 3
num_samples = prnd_seq_len * 100
# --------------------------------------------------------------

rtl = RTL_utils()
sig = rtl.rtl_read(
	freq=center_freq, 
	rate=rx_rate, 
	gain=rx_gain,
	num_samples=num_samples)

# F = np.abs(fft(sig))
# k = np.argmax(F)
# exp = np.exp(1j * np.arange(len(sig)) * 2 * np.pi * k / len(sig))
# sig -= np.dot(sig, exp.conj()) * exp / len(sig)

# n_avg = 10
# len_desired = ceil(len(sig) / n_avg) * n_avg
# sig = np.concatenate((sig, np.zeros(len_desired - len(sig), dtype=sig.dtype)))
# sig = np.reshape(sig, (n_avg, -1))

# F = np.abs(fftshift(fft(sig, axis=1), axes=1))
# f = np.linspace(-rx_rate/2e3, rx_rate/2e3, sig.shape[1])
# plt.ylabel('Magnitude')
# plt.xlabel('Frequency (kHz)')
# plt.plot(f, np.mean(F, axis=0))
# plt.show()

# F = np.abs(fft(sig))
# plt.plot(F)
# plt.show()
# k = np.argmax(F)
# sig *= np.exp(-1j * np.arange(len(sig)) * 2 * np.pi * k / len(sig))
# plt.subplot(211)
# plt.plot(sig.real)
# plt.subplot(212)
# plt.plot(sig.imag)
# plt.show()

# M = 1
# sig = sig_utils.my_resample(sig, 1, M)
# rx_rate = rx_rate // M

rtl.view_spectrum(sig, center_freq, rx_rate, 1)

prnd_seq = sig_utils.prnd_gen(seq_len=prnd_seq_len, type=prnd_type, mode=prnd_mode, seed=prnd_seed)


dec= SSM_decoder(rx_rate, prnd_seq, pt_bw=1e6)

# chop = prnd_seq_len // (10 * M)
chop = prnd_seq_len
# # print(chop, chop * M)
est = dec.motion_estimate_iq(sig, chop=chop, mode='RSSM', normalize=False)
t = np.arange(len(est)) * chop / rx_rate

plt.plot(t, est)
plt.show()

# exp = np.exp(-1j * np.arange(prnd_seq_len) * dec.omega)
# cor = sig_utils.my_cor(exp * sig, prnd_seq)
# ind = np.argmax(np.abs(cor))
# prnd_seq = np.roll(prnd_seq, -ind)

# plt.plot(np.real(exp * sig * prnd_seq))
# plt.show()