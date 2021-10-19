import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from ..utils import sig_utils, UHD_utils

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------


# --------------- CHANGE SEQUENCE PARAMTERS HERE ---------------
seq_id = '/scans/cory_test'
center_freq = 127.7e6
tx_rate = 500e3
tx_gain = 50
prnd_seq_len = 2 ** 7
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
iq_sig = sig_utils.prnd_gen(seq_len=prnd_seq_len, type=prnd_type, mode=prnd_mode, seed=prnd_seed)

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
	s += key + ':\n'
	s += repr(seq_data[key])
	s += '\n\n'
with open(uhd.PY_DIR +  seq_id + '.txt', 'w') as f:
	f.write(s)

# Transmit iq signal infinitely
uhd.sdr_write(
	iq_sig=iq_sig,
	freq=center_freq,
	rate=tx_rate,
	gain=tx_gain,
	repeat=True)
