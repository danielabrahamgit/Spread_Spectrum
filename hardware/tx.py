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
# UHD_DIRECTORY = '/opt/local/share/uhd/examples/'
# --------------------------------------------------------------
# --------------- CHANGE SEQUENCE PARAMTERS HERE ---------------
seq_id = '/scans/cory_sine_5hz'
center_freq = 127697899
tx_rate = 200e3
tx_gain = 0
prnd_seq_len = 20
prnd_type = 'bern'
prnd_mode = 'real'
prnd_seed = 10
L = 1
# --------------------------------------------------------------
# Make instance of UHD_utils class
uhd = UHD_utils(UHD_DIRECTORY)
# SSM iq_sig gen
n = np.arange(prnd_seq_len)
prnd_seq = sig_utils.prnd_gen(
            seq_len=prnd_seq_len, 
            type=prnd_type, 
            mode=prnd_mode, 
            seed=prnd_seed
)
N_repeats = 1
iq_sig = np.tile(prnd_seq, N_repeats)
# Transmit iq signal infinitely

print(iq_sig[:10])

uhd.uhd_write(
    iq_sig=iq_sig,
    freq=center_freq,
    rate=tx_rate,
    gain=tx_gain,
    repeat=True,
    arg='3215B94',
    clk=True
)
# # Store all parameters in dictionary (Eventually text file)
# seq_data = {}
# seq_data['prnd_seq_len'] = prnd_seq_len
# seq_data['prnd_type'] = prnd_type
# seq_data['prnd_mode'] = prnd_mode
# seq_data['prnd_seed'] = prnd_seed
# seq_data['upsample_factor'] = L
# seq_data['center_freq'] = center_freq
# seq_data['tx_rate'] = tx_rate
# seq_data['tx_gain'] = tx_gain
# s = ''
# for key in seq_data.keys():
#     s += key + ': '
#     s += str(seq_data[key])
#     s += '\n\n'
# with open(uhd.PY_DIR +  seq_id + '.txt', 'w') as f:
#     f.write(s)