import sys
import os

import argparse
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from utils.MR_utils import MR_utils
from utils.SSM_decoder import SSM_decoder
from PIL import Image

# Argument parser
parser = argparse.ArgumentParser(description='Standard and Spread Spectrum Pilot Tone Simulator. \
	See https://github.com/danielabrahamgit/Spread_Spectrum for documentation.')
parser.add_argument('-pt_fc', metavar='pt_fc', type=float, default=0,
					help='The Pilot Tone transmission frequency \
							(assumes no SSM) (kHz). Default = 0kHz')
parser.add_argument('--ssm', action='store_true',
					help='Do you want to enable spread spectrum? \
						Default = False')
parser.add_argument('--robust', action='store_true',
					help='Uses the brute-force robust SSM technique. \
						Default = False')
parser.add_argument('--phase_rnd', action='store_true',
					help='Do you want to enable uncertainty in the phase? \
						Default = False')
parser.add_argument('-tr', metavar='tr', type=float, default=34,
					help='Time repetition (TR): Time between readouts (ms). \
						Default = 34ms')
parser.add_argument('-pt_bw', metavar='pt_bw', type=float, default=250,
					help='Bandwidth of Pilot Tone: Range of frequencies in imaging band (kHz). \
						Default = imaging bandwidth')
parser.add_argument('-im_bw', metavar='im_bw', type=float, default=250,
					help='Bandwidth of Pilot Tone: Range of frequencies in imaging band (kHz). \
						Default = 250kHz')
parser.add_argument('-pt_amp', metavar='pt_amp', type=float, default=20,
					help='Dimensionless unit of amplitude for the pilot tone. \
						Default = 20')
parser.add_argument('-pt_uncert', metavar='pt_uncert', type=float, default=5,
					help='Uncertanty of PT center frequency (Hz). \
						Default = 5 Hz')
args = parser.parse_args()

# Uncertanty in TR
if args.phase_rnd:
	phase_rnd = 0.01
else:
	phase_rnd = 0

# Pilot tone frequency
fpt = args.pt_fc * 1e3

# Load MR image
# im = np.array(Image.open('images/brain.png'))
im = np.load('images/brain.npz')['im']

# Initialize MR object with the parameters below
mr = MR_utils(tr=args.tr * 1e-3, bwpp=args.im_bw * 1e3/im.shape[1], pt_bw=args.pt_bw * 1e3, robust=args.robust)

# Load Data into MR object
mr.load_image(im)

# Motion modulation signal
t = np.arange(0, mr.ksp.shape[0], mr.TR)
mod = (1 + 0.5 * np.sin(2 * np.pi * t ))

# Uncertantiy in hertz
fpt_uncert = args.pt_uncert
fpt_actual = fpt + np.random.uniform(-fpt_uncert, fpt_uncert)
fpt_actual = -2.42e3
print(f'Doppler Actual = {fpt_actual/1e3} (kHz)')

# Spread spectrum modulation PRN sequence
if args.ssm:
	seq_len = mr.ksp.shape[1]
	mr.prnd_seq_gen(seq_len=seq_len, type='bern', mode='real', seed=1)

# Add Pilot tone (with modulation) and extract motion + image
a, b = mr.add_PT(fpt_actual, pt_amp=args.pt_amp, phase_uncert=phase_rnd, modulation=mod)
print(mr.true_rnd[0][:10])

# Plot motion estimates
ssm_dec = SSM_decoder(
				mr_bw=args.im_bw * 1e3, 
				prnd_seq=mr.prnd_seq, 
				pt_fc=args.pt_fc * 1e3, 
				pt_bw=args.pt_bw * 1e3,
				doppler_range=fpt_uncert
)
est_motion = ssm_dec.motion_estimate_ksp(mr.ksp, mode='RSSM', normalize=True)
true_motion = mr.true_motion

# Print L1 and L2 errors
print('M(abs)E:', np.sum(np.abs(est_motion - true_motion)) / mr.ksp.shape[0])
print('MSE:', np.sum((est_motion - true_motion) ** 2) / mr.ksp.shape[0])

# Show PSNR
print(f'SNR(dB): {10 * np.log10(mr.P_ksp / mr.P_pt)}')

# Display motion estimate
plt.title('Pilot Tone Motion Estimate')
plt.xlabel('Phase Encode #')
plt.ylabel('PT Magnitude')
plt.plot(est_motion, label='Estimated')
plt.plot(true_motion, label='True Modulation', color='r')
plt.legend()

# Show eveything
mr.MRshow(drng=1e-6, log_ksp=True, log_ro=True, log_im=False, select=3)