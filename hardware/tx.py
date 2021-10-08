import matplotlib.pyplot as plt
import numpy as np
from UHD_utils import UHD_utils
from scipy import signal
from PT_utils import *

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# Remebering the sequence
seq_id = '/scans/cory_long'
seq_data = {}

# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)

# uhd parameters
center_freq = 127.7e6
tx_rate = 500e3
gain = 50
seq_data['center_freq'] = center_freq
seq_data['tx_rate'] = tx_rate
seq_data['gain'] = gain

# SSM Paramters
prnd_len = 2 ** 7

# SSM iq_sig gen
iq_sig = gen_prnd(prnd_len)
seq_data['rnd_seq'] = iq_sig

# Resample
L = 1
# iq_sig_new_rate = np.zeros(len(iq_sig) * L)
# iq_sig_new_rate[::L] = iq_sig
# h_lpf = signal.firwin(129, tx_rate/2, fs=tx_rate * L)
# iq_sig_new_rate = np.convolve(iq_sig_new_rate, h_lpf, mode='same')

seq_data['upsample_factor'] = L

# Save params
s = ''
for key in seq_data.keys():
	s += key + ':\n'
	s += repr(seq_data[key])
	s += '\n\n'
with open(uhd.PY_DIR +  seq_id + '.txt', 'w') as f:
	f.write(s)


filename=seq_id + '.dat'

# Transmit iq signal infinitely
if L != 1:
	uhd.sdr_write(
		iq_sig=iq_sig_new_rate,
		freq=center_freq,
		rate=tx_rate * L,
		gain=gain,
		file=filename,
		repeat=True)
else:
	uhd.sdr_write(
		iq_sig=iq_sig,
		freq=center_freq,
		rate=tx_rate,
		gain=gain,
		file=filename,
		repeat=True)
