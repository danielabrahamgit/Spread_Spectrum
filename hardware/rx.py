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
center_freq = 127697458
rx_rate = 500e3
rx_gain = 0
prnd_seq_len = 22000
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
<<<<<<< HEAD
num_samples = prnd_seq_len * 100
=======
read_time = 20
num_samples = rx_rate * read_time
M = 1
>>>>>>> ab0dbc4f9878aa0519978dc9986fd31da2917aec
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
			rate=rx_rate,
			gain=rx_gain,
<<<<<<< HEAD
			duration=num_samples / rx_rate,
			arg='3215B78',
			use_sdr=False,
			file='uhd_iq/write.dat'
)

print(iq_sig[:10])
=======
			duration=read_time,
			arg='3215B78',
			use_sdr=True
)

dec = SSM_decoder(
			mr_bw=rx_rate, 
			prnd_seq=prnd_seq, 
			pt_fc=0, 
			pt_bw=rx_rate // 2,
			doppler_range=1e3
)

est = dec.motion_estimate_iq(
				iq_sig=iq_sig, 
				mode='RSSM',
				chop=256
)
plt.plot(np.arange(len(est)) * 256 / rx_rate, np.abs(est))
plt.show()
>>>>>>> ab0dbc4f9878aa0519978dc9986fd31da2917aec
