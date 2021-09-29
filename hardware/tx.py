import matplotlib.pyplot as plt
import numpy as np
from UHD_utils import UHD_utils
from scipy import signal
from PT_utils import *

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)

# uhd parameters
center_freq = 127e6
tx_rate = 500e3

# SSM Paramters
prnd_len = 1001//2

# SSM iq_sig gen
iq_sig = gen_prnd(prnd_len)
# n = np.arange(prnd_len)
# iq_sig = np.cos(2 * np.pi * tx_rate/4 * n / tx_rate)

# Resample
L = 2
iq_sig_new_rate = np.zeros(len(iq_sig) * L)
iq_sig_new_rate[::L] = iq_sig
h_lpf = signal.firwin(129, tx_rate/2, fs=tx_rate * L)
iq_sig_new_rate = np.convolve(iq_sig_new_rate, h_lpf, mode='same')


# Transmit iq signal infinitely
uhd.sdr_write(
	iq_sig=iq_sig_new_rate,
	freq=center_freq,
	rate=tx_rate * L,
	gain=40,
	repeat=True)

