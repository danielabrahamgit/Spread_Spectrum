import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import MR_utils
from PIL import Image

# Argument parser
parser = argparse.ArgumentParser(description='Standard and Spread Spectrum Pilot Tone Simulator. \
	See https://github.com/danielabrahamgit/Spread_Spectrum for documentation.')
parser.add_argument('-fpt', metavar='fpt', type=float, default=120e3,
					help='The standard pilot tone frequency \
							(assumes no SSM) (kHz). Default = 120kHz')
parser.add_argument('--ssm', action='store_true',
					help='Do you want to enable spread spectrum? \
						Default = False')
parser.add_argument('--tr_rnd', action='store_true',
					help='Do you want to enable uncertainty in TR? \
						Default = False')
parser.add_argument('-tr', metavar='tr', type=float, default=34e-3,
					help='Time repetition (TR): Time between readouts (ms). \
						Default = 34ms')
parser.add_argument('-bw', metavar='bw', type=float, default=250e3,
					help='Bandwidth: Range of frequencies in imaging band (kHz). \
						Default = 250kHz')
parser.add_argument('-fc', metavar='fc', type=float, default=127.8e6,
					help='Center frequency of scanner (MHz). \
						Default = 127.8MHz')
args = parser.parse_args()


# Uncertanty in TR
if args.tr_rnd:
	tr_uncert = 0.001
else:
	tr_uncert = 0
# Pilot tone frequency
if args.ssm:
	fpt = 0
else:
	fpt = args.fpt

# Load MR image
# im = np.array(Image.open('images/brain.png'))
im = np.load('images/brain.npz')['im']

# Initialize MR object with the parameters below
mr = MR_utils(tr=args.tr, bwpp=args.bw/max(im.shape), fc=args.fc)

# Load Data into MR object
mr.load_image(im)

# Motion modulation signal
def mod(t):
	return 5 * (2 + np.sin(2 * np.pi * t / (50 * mr.TR)))

# Spread spectrum modulation PRN sequence
if args.ssm:
	mr.prnd_seq_gen(p=0.5, seq_len=mr.ksp.shape[1])

ws = np.linspace(-1, 1, len(mr.prnd_seq))
plt.plot(ws, np.abs(np.fft.fftshift(np.fft.fft(mr.prnd_seq))))
plt.ylabel('LFSR FFT Magnitude')
plt.xlabel(r'$\frac{\omega}{\pi}$')
plt.show()
quit()

# Get k-sapce std before adding the PT
ksp_std = mr.get_ksp_std()

# Add Pilot tone (with modulation) and extract motion + image
a, b = mr.add_PT(fpt, tr_uncert=tr_uncert, modulation=mod)

# Plot motion estimates
motion = mr.motion_extract(fpt=fpt)
true_motion = mr.true_motion

# Showcase effect of k-space standard deviation 
# on the inner product method
if args.ssm:
	plt.subplot(2,1,1)
	plt.title('Standard Deviation of Each Readout')
	plt.xlabel('Phase Encode #')
	plt.ylabel('$\sigma$')
	plt.plot(ksp_std)
	plt.subplot(2,1,2)
	plt.title('Inner Product Estimate')
	plt.xlabel('Phase Encode #')
	plt.ylabel('PT Magnitude')
	plt.plot(np.abs(motion), label='Inner Product Estimate')
	plt.plot(np.abs(true_motion), label='True Modulation', color='r')
	plt.legend()
else:
	plt.title('Pilot Tone Motion Estimate')
	plt.xlabel('Phase Encode #')
	plt.ylabel('PT Magnitude')
	plt.plot(np.abs(motion), label='Estimated')
	plt.plot(np.abs(true_motion), label='True Modulation', color='r')
	plt.legend()

# Show eveything
mr.MRshow(drng=1e-6, log_ksp=True, log_ro=True, log_im=False)