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
seq_id = '/scans/cory_discovery.txt'
center_freq = 127697458
tx_rate = 250e3
tx_gain = 5
prnd_seq_len = 22000
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
# --------------------------------------------------------------

# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)

# SSM iq_sig gen
prnd_seq = sig_utils.prnd_gen(
			seq_len=prnd_seq_len, 
			type=prnd_type, 
			mode=prnd_mode, 
			seed=prnd_seed
)

# --------------- STANDARD PT CODE ---------------
# Frequency offset from center_freq
iq_sig_len = 25000
f_offset = 100e3
n = np.arange(iq_sig_len)
iq_sig = np.exp(2j * np.pi * (f_offset) * n / tx_rate)
# ------------------------------------------------

# N_repeats is the number of time the iq_sig will repeat 
# when stored in the IQ file. 
N_repeats = 20
iq_sig = np.tile(iq_sig, N_repeats)

iq_sig = iq_sig / (np.max(np.abs(iq_sig)))
iq_sig = iq_sig.astype(np.complex64)

# Transmit iq signal infinitely
uhd.uhd_write(
	iq_sig=iq_sig,
	freq=center_freq,
	rate=tx_rate,
	gain=tx_gain,
	repeat=True,
	arg='3215B94'
)
