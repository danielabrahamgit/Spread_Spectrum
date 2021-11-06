import sys
import os
<<<<<<< HEAD
from numpy.testing._private.utils import IgnoreException
sys.path.append(os.path.abspath('../'))
=======
sys.path.append(os.path.abspath('../'))

>>>>>>> ab0dbc4f9878aa0519978dc9986fd31da2917aec
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift, fft
from scipy import signal
from utils.sig_utils import sig_utils
from utils.SDR_utils import UHD_utils
from utils.SSM_decoder import SSM_decoder
<<<<<<< HEAD
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
=======

# SA input arguments via argparse
import argparse

def main(argv):
    parser = argparse.ArgumentParser(description="Transmit pure tone via IQ file")
    parser.add_argument("-o", "--outdir", default="/scans",
                        help="Output file directory to write file to ")
    parser.add_argument("-g", "--gain", default=90,
                        help="Transmit gain")
    parser.add_argument("-f", "--freq", default=2.4e9,
                        help="Center frequency (default 2.4e9)")
    parser.add_argument("-r", "--rate", default=0.5e6,
                        help="Transmit rate (default 0.5e6)")
    parser.add_argument("-s", "--serial", default='3215B94',
                        help="Serial number (default '3215B94')")
    parser.add_argument("-i", "--id", default='scan',
                        help="Sequence id for scan parameters")

    args = parser.parse_args()

    # ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
    UHD_DIRECTORY = '/opt/local/share/uhd/examples/'

    # Make instance of UHD_utils class
    uhd = UHD_utils(UHD_DIRECTORY)

    # --------------- STANDARD PT CODE ---------------
    # Frequency offset from center_freq
    iq_sig_len = 25000
    f_offset = 100e3
    ampl = 0.7
    n = np.arange(iq_sig_len)
    iq_sig = np.exp(2j * np.pi * (f_offset) * n / float(args.rate))
    # ------------------------------------------------

    # N_repeats is the number of time the iq_sig will repeat 
    # when stored in the IQ file. 
    N_repeats = 500
    iq_sig = np.tile(iq_sig, N_repeats)

    iq_sig = ampl * iq_sig / (np.max(np.abs(iq_sig)))
    iq_sig = iq_sig.astype(np.complex64)

    uhd.save_sequence_params(args.id, args.freq, args.rate, args.gain)

    uhd.uhd_write(
        iq_sig=iq_sig,
        freq=args.freq,
        rate=args.rate,
        gain=args.gain,
        repeat=True,
        arg=args.serial,
        file=args.id + '.dat'
        )


if __name__ == '__main__':
    main(sys.argv[1:])

>>>>>>> ab0dbc4f9878aa0519978dc9986fd31da2917aec
