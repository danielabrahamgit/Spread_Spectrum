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
center_freq = 150e6
tx_rate = 1e6
tx_gain = 5
prnd_seq_len = 256
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
L = 1
# --------------------------------------------------------------

print(f'Time={prnd_seq_len / tx_rate}')

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
prnd_seq = sig_utils.prnd_gen(
			seq_len=prnd_seq_len, 
			type=prnd_type, 
			mode=prnd_mode, 
			seed=prnd_seed
)


N = 100
A = np.linspace(0, 1, N)
# A = 2 + np.sin(np.linspace(0, 2 * np.pi, N))
A = np.repeat(A, prnd_seq_len)
A = A / np.max(np.abs(A))
iq_sig = np.tile(prnd_seq, N) * A
# iq_sig = A * np.exp(2j * np.pi * (100e3) * np.arange(len(A)) / tx_rate)

# IQ SIG BANK
# iq_sig = np.exp(2j * np.pi * (100e3) * n / tx_rate)
# iq_sig = signal.square(2 * np.pi * 100e3 * n / tx_rate)
# iq_sig = np.ones(prnd_seq_len, dtype=np.complex64)
# iq_sig = np.exp(-0.001 * n)


# SQUARE/TRIANGLE FREQ GEN
wc = 0.2
# iq_sig = signal.firwin(prnd_seq_len, wc, fs=2).astype(np.complex64)
# iq_sig = iq_sig ** 2 / wc

# SHIFT IN FREQUENCY
iq_sig = iq_sig.astype(np.complex64)
w0 = 0e3
iq_sig *= np.exp(1j * np.arange(len(iq_sig)) * 2 * np.pi * (w0) / tx_rate)
# iq_sig *= np.exp(1j * np.arange(len(iq_sig)) * 2 * np.pi * (-w0) / tx_rate)

dec = SSM_decoder(
			mr_bw=tx_rate, 
			prnd_seq=prnd_seq, 
			pt_fc=0, 
			pt_bw=tx_rate,
			doppler_range=10e3
)

# est = dec.motion_estimate_iq(iq_sig, chop=prnd_seq_len, normalize=False)
# plt.plot(est)
# plt.show()

# cor = sig_utils.my_cor(prnd_seq, iq_sig)
# plt.plot(np.real(cor))
# plt.show()

f = np.linspace(-tx_rate/2e3, tx_rate/2e3, len(iq_sig))
plt.ylabel('Magnitude')
plt.xlabel('Frequency (kHz)')
plt.plot(f, np.abs(fftshift(fft(iq_sig))))
plt.show()



# Resample If needed
if L != 1:
	iq_sig = signal.resample_poly(iq_sig, L, 1)
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
	repeat=True,
	arg='3215B94'
)
