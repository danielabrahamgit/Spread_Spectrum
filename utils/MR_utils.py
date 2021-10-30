import sys
import os

from numpy.fft import fftshift, fft
sys.path.append(os.path.abspath('../'))

import numpy as np
import matplotlib.pyplot as plt
from utils.sig_utils import sig_utils
from scipy import signal

class MR_utils:
	# Set all global constants on initialization
	def __init__(self, tr=0, bwpp=0, pt_bw=None, robust=False):	
		# Timing parameters
		self.TR  = tr
		self.BWPP = bwpp

		# Which SSM technique?
		self.robust = robust

		# Frequencies
		self.fs = None # Will be calculated soon
		self.fs_pt = pt_bw

		# PT and image power levels
		self.P_pt = -1
		self.P_ksp = -1

		# image and kspace of that image
		self.img = None
		self.ksp = None

		# For debuging
		self.ksp_og = None
		self.true_rnd = []

		# PRND sequence for pilot tone
		self.prnd_seq = None

		# True modulation
		self.true_motion = []
	
	def reset_states(self):
		# PT and image power levels
		self.P_pt = -1
		self.P_ksp = -1

		# For debuging
		self.ksp_og = None
		self.true_inds = []

		# PRND sequence for pilot tone
		self.prnd_seq = None

		# True modulation
		self.true_motion = []
	
	# Load an MR image (image space)
	def load_image(self, img):
		self.img = img
		self.ksp = sig_utils.fft2c(img)
		self.fs = self.ksp.shape[1] * self.BWPP
		if self.fs_pt is None:
			self.fs_pt = self.fs
	
	# Load a K-space image
	def load_kspace(self, ksp):
		self.img = sig_utils.ifft2c(ksp)
		self.ksp = ksp
		self.fs = self.ksp.shape[1] * self.BWPP
		if self.fs_pt is None:
			self.fs_pt = self.fs
	
	# Display image and kspace
	def MRshow(self, drng=1e-6, log_ksp=True, log_ro=False, log_im=False, title='', select=7):
		# Check that we have images
		if self.img is None or self.ksp is None:
			print('Load image first')
			return
		
		if self.prnd_seq is not None:

			# KSPACE DISPLAY
			if select & 1 > 0:
				plt.figure()
				plt.title('K-Space')
				if log_ksp:
					plt.imshow(np.log(np.abs(self.ksp) + drng), cmap='gray')
				else:
					plt.imshow(np.abs(self.ksp), cmap='gray')
			
			# IMAGE DISPLAY
			if select & 2 > 0:
				plt.figure()
				plt.title('Acquired Image')
				self.img[self.img.shape[0]//2, self.img.shape[1]//2] = 0
				if log_im:
					plt.imshow(np.log10(np.abs(self.img)), cmap='gray')
				else:
					plt.imshow(np.abs(self.img), cmap='gray')
			
			# FFT READOUT DISPLAY
			if select & 4 > 0:
				plt.figure()
				fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
				plt.title(f'FFT Along Readout')
				plt.xlabel(r'$\frac{\omega}{\pi}$')
				plt.ylabel('Phase Encode #')
				if log_ro:
					plt.imshow(np.log10(np.abs(fft_readout)), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
				else:
					plt.imshow(np.abs(fft_readout), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
		plt.show()

	# Adds a pure pilot tone to our kspace data
	def add_PT(self, freq, pt_amp=30, phase_uncert=0, modulation=None):
		# Number of phase encodes and readouts
		n_pe, n_ro = self.ksp.shape

		# Keep sequence of ones if no random sequence is selected
		if self.prnd_seq is None:
			prnd_seq = np.ones(self.ksp.shape[1] * 2)
		else:
			prnd_seq = self.prnd_seq.copy()
		
		# If no modulation, just make a function that returns 1
		if modulation is None:
			modulation = np.ones(n_pe)

		# For debugging
		self.ksp_og = self.ksp.copy()
		
		# Time of readout in seconds
		t_readout = 1 / self.BWPP

		# Time axis for PT device
		N_pt_ro = int(t_readout * self.fs_pt)
		n = np.arange(N_pt_ro)

		# true indices variable
		true_ind = 0

		# Add pilot tone to kspace data for readout
		for pe in range(n_pe):
			# Factor in uncertanty in TR
			if phase_uncert != 0:
				TR = self.TR * (1 + (((2 * np.random.rand()) - 1) * phase_uncert))
			else:
				TR = self.TR
			
			# Number of samples added from previous readout
			samples_added = int(TR * self.fs_pt) % N_pt_ro 
			# samples_added = np.random.randint(0, N_pt_ro)

			# True shift of random sequence (for debugging mainly)
			true_ind = (true_ind + samples_added) % N_pt_ro
			
			# Adjust the pseudo random sequence
			prnd_seq = np.roll(prnd_seq, -samples_added)
			self.true_rnd.append(prnd_seq[:N_pt_ro])

			# Device signal is then modulation * pilot tone * rnd sequence
			# pt_sig_device = pt_amp * modulation[pe] * np.exp(2j*np.pi*freq*(n + samples_added) / self.fs_pt) * prnd_seq[:N_pt_ro]
			phase_rnd = 1 + 0 * np.exp(1j * np.random.uniform(0, 2 * np.pi))
			pt_sig_device = phase_rnd * pt_amp * modulation[pe] * np.exp(2j*np.pi*freq *n / self.fs_pt) * prnd_seq[:N_pt_ro]

			# The scanner receivess a resampled version of the original PT signal due to 
			# a mismatch in the sacnner BW and the PT device BW
			pt_sig_scanner = signal.resample_poly(pt_sig_device, n_ro, N_pt_ro)
			
			# plt.plot(np.abs(fftshift(fft(pt_sig_scanner))))
			# plt.show()

			# Keep track of the power levels of the pilot tone and raw readout speraately
			self.P_pt += np.sum(np.abs(pt_sig_scanner) ** 2)
			self.P_ksp += np.sum(np.abs(self.ksp[pe,:]) ** 2)

			# Add pilot tone to the scanner
			self.ksp[pe,:] += pt_sig_scanner

			# Keep track of the true modulation signal, for later
			self.true_motion.append(modulation[pe])
		
		# Recalculate image
		self.img = sig_utils.ifft2c(self.ksp)

		# Normalize the true motion
		self.true_motion = sig_utils.normalize(np.array(self.true_motion))

		# Calculate pilot tone location
		val = freq / self.fs
		beta = np.round(val) - val
		b = np.round(n_ro/2 + beta * n_ro)

		val = freq * self.TR
		alpha = np.round(val) - val
		a = np.round(n_pe/2 + alpha * n_pe)

		return a, b

	# Generates a single sequence that will repeat itself
	def prnd_seq_gen(self, seq_len=None, type='bern', mode='real', seed=None):
		if seq_len is None:
			seq_len = self.ksp.shape[1]
		
		self.prnd_seq = sig_utils.prnd_gen(seq_len, type, mode, seed)
			
		N_pt_ro = int(self.ksp.shape[1] * self.fs_pt / self.fs)
		if seq_len < N_pt_ro:
			self.prnd_seq = np.tile(self.prnd_seq, int(np.ceil(N_pt_ro / len(self.prnd_seq))))