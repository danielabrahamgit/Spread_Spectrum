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

