import argparse
from SSM_decoder import SSM_decoder
import numpy as np
import matplotlib.pyplot as plt
from utils import MR_utils, sig_utils
from PIL import Image

# Argument parser
parser = argparse.ArgumentParser(description='Standard and Spread Spectrum Pilot Tone Simulator. \
	See https://github.com/danielabrahamgit/Spread_Spectrum for documentation.')
parser.add_argument('-fpt', metavar='fpt', type=float, default=0,
					help='The standard pilot tone frequency \
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
args = parser.parse_args()

# Uncertanty in TR
if args.phase_rnd:
	phase_rnd = 0.0022
else:
	phase_rnd = 0
# Pilot tone frequency
fpt = args.fpt * 1e3

# Load MR image
# im = np.array(Image.open('images/brain.png'))
im = np.load('../images/brain.npz')['im']

# Initialize MR object with the parameters below
mr = MR_utils(tr=args.tr * 1e-3, bwpp=args.im_bw * 1e3/im.shape[1], pt_bw=args.pt_bw * 1e3, robust=args.robust)

# Load Data into MR object
mr.load_image(im)

# Motion modulation signal
def mod(t):
	return args.pt_amp * (1 + 0.5 * np.sin(2 * np.pi * t / (50 * mr.TR)))

# Spread spectrum modulation PRN sequence
if args.ssm:
	mr.prnd_seq_gen(p=0.5, seq_len=mr.ksp.shape[1] * 2)

# Get k-sapce std before adding the PT
ksp_std = mr.get_ksp_std()

# Add Pilot tone (with modulation) and extract motion + image
a, b = mr.add_PT(fpt, phase_uncert=phase_rnd, modulation=mod)

# Plot motion estimates
ssm_dec = SSM_decoder(args.im_bw, mr.prnd_seq, pt_fc=args.fpt, pt_bw=args.pt_bw)
motion, ind = ssm_dec.motion_estimate(mr.ksp, mode='ballpark')
true_motion = mr.true_motion

# Print L1 and L2 errors
print('M(abs)E:', np.sum(np.abs(motion - true_motion)) / mr.ksp.shape[0])
print('MSE:', np.sum((motion - true_motion) ** 2) / mr.ksp.shape[0])

# Show PSNR
print(f'SNR(dB): {10 * np.log10(mr.P_ksp / mr.P_pt)}')

# Display motion estimate
plt.title('Pilot Tone Motion Estimate')
plt.xlabel('Phase Encode #')
plt.ylabel('PT Magnitude')
plt.plot(motion, label='Estimated')
plt.plot(true_motion, label='True Modulation', color='r')
plt.legend()

# Show eveything
mr.MRshow(drng=1e-6, log_ksp=True, log_ro=True, log_im=False)