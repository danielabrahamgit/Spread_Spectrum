import matplotlib.pyplot as plt
import numpy as np
from UHD_utils import UHD_utils
from scipy import signal

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)

# uhd parameters
freq = 677.887e6
rate = 100e3
duration = 0.05

# iq_sig gen
f = 220e3
N = int(duration * rate)
n = np.arange(N)
iq_sig = 1 - 2 * np.random.randint(0, 2, N)

# Resample to higher rate in order to impliment digital LPF
L = 2
iq_up = np.zeros(L * len(iq_sig))
iq_up[::L] = iq_sig
h = signal.firwin(10, rate/2, fs=rate * L)
iq_sig = np.convolve(iq_up, h, mode='valid')

# Transmit iq signal infinitely
uhd.sdr_write(
	iq_sig=iq_sig,
	freq=freq,
	rate=rate * L,
	gain=30,
	repeat=True)

