import sys
import os

from numpy.testing._private.utils import IgnoreException
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
center_freq = 2.4e9
tx_rate = 1e6
tx_gain = 40
# --------------------------------------------------------------

# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)

# Tone at frequency 2.4HGz
n = np.arange(1000)
A = 20 * (2 + np.sin(np.linspace(0, 2*np.pi, len(n))))
f0 = 100e3
iq_sig = A
plt.plot(iq_sig.real)
plt.show()

n_fft = 1024
f = np.linspace(center_freq -tx_rate/2, center_freq + tx_rate/2, n_fft)
plt.plot(f, np.abs(fftshift(fft(iq_sig, n_fft))))
plt.show()
quit()

# Transmit iq signal infinitely
uhd.uhd_write(
	iq_sig=iq_sig,
	freq=2.4e9,
	rate=tx_rate,
	gain=tx_gain,
	repeat=True,
	arg='3215B94'
)
