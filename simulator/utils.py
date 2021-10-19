import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import signal
from numpy.fft import fft, fftshift

class sig_utils:

	# Simulates resampling twice 
	def pt_to_mr_to_pt(pt_sig, up, down):
		mr_sig = sig_utils.my_resample(pt_sig, down, up)
		return sig_utils.my_resample(mr_sig, up, down)

	def auto_cor_mat(X):
		X_fft = np.fft.fft(X, axis=1)
		return np.fft.ifft(X_fft * X_fft.conj(), axis=1).real


	# My version of resample
	def my_resample(x, up, down, ntaps=129):
		gcd = np.gcd(int(up), int(down))
		up = int(up) // gcd
		down = int(down) // gcd
		N = len(x)
		x_up = np.zeros(N * up, dtype=x.dtype)
		x_up[::up] = x
		h_lp = signal.firwin(ntaps, min(1/up, 1/down), fs=2) * up
		x_up_lp = np.convolve(x_up, h_lp, mode='same')
		return x_up_lp[::down]
		

	# Normalizes a signal
	def normalize(x):
		mu = np.mean(x)
		sig = np.std(x)
		if sig > 0:
			return (x - mu) / sig
		else:
			return x - mu

	# Sparsifying threshold
	def SoftThresh(y, lambd):
		return (0 + (np.abs(y) - lambd) > 0) * y * (np.abs(y) - lambd) / (np.abs(y) + 1e-8)

	# My circular cross correlation
	def my_cor(x, y):
		N = max(len(x), len(y))
		return np.fft.ifft(np.fft.fft(x, n=N) * np.fft.fft(y, n=N).conj())

	# 2DFT and inverse 
	def fft2c(f, shp=None):
		if shp == None:
			shp = f.shape
		return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f), s=shp))
	
	def ifft2c(F, shp=None):
		if shp == None:
			shp = F.shape
		return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F), s=shp))	

	# Generates purely orthogonal codes
	def hadamard(n):
		try:
			assert (n & (n-1) == 0) and n != 0
		except:
			print("'n' needs to be a power of 2")
		
		if n == 1:
			return 1
		
		h_half = sig_utils.hadamard(n//2)
		h_top = np.hstack((h_half, h_half))
		h_bot = np.hstack((h_half, -h_half))

		return np.vstack((h_top, h_bot))

	# Helper for lfsr_gen
	def hex_to_bits(hex_int):
		bits = []
		for i in range(16):
			bits.append(2 * (hex_int & 1) - 1)
			hex_int = hex_int >> 1
		return bits

	# returns matrix with PRND bit sequence in rows
	def lfsr_gen(seq_len, lfsr=0xACE1):
		nums = []
		for i in range(ceil(seq_len / 16)):
			bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1
			lfsr = (lfsr >> 1) | (bit << 15)
			nums += sig_utils.hex_to_bits(lfsr)
		return np.array(nums)


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
		self.true_inds = []

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
	def MRshow(self, drng=1e-6, log_ksp=True, log_ro=False, log_im=False, title=''):
		# Check that we have images
		if self.img is None or self.ksp is None:
			print('Load image first')
			return
		
		if self.prnd_seq is not None:
			# Generate subplots
			# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))

			# Display kspace
			plt.figure()
			plt.title('K-Space')
			if log_ksp:
				plt.imshow(np.log(np.abs(self.ksp) + drng), cmap='gray')
			else:
				plt.imshow(np.abs(self.ksp), cmap='gray')
			
			# Display image (top right)
			plt.figure()
			plt.title('Acquired Image')
			self.img[self.img.shape[0]//2, self.img.shape[1]//2] = 0
			if log_im:
				plt.imshow(np.log10(np.abs(self.img)), cmap='gray')
			else:
				plt.imshow(np.abs(self.img), cmap='gray')
			
			# fig.suptitle(title)

			# # Display fft along readout (bottom left)
			plt.figure()
			fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
			plt.title(f'FFT Along Readout')
			plt.xlabel(r'$\frac{\omega}{\pi}$')
			plt.ylabel('Phase Encode #')
			if log_ro:
				plt.imshow(np.log10(np.abs(fft_readout)), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
			else:
				plt.imshow(np.abs(fft_readout), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
			
			# # Display correlated fft along raeadout (bottom right)
			plt.figure()
			plt.title('Auto-Correlation of RND Sequence')
			auto = sig_utils.my_cor(self.prnd_seq, self.prnd_seq).real
			inds = np.arange(-len(auto)//2, len(auto)//2) 
			plt.plot(inds, np.fft.fftshift(auto))
		# Regular Pilot Tone
		else:
			# Generate subplots
			fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(10, 6))

			# Display kspace (top)
			ax1.set_title('K-Space')
			if log_ksp:
				ax1.imshow(np.log(np.abs(self.ksp) + drng), cmap='gray')
			else:
				ax1.imshow(np.abs(self.ksp), cmap='gray')
			
			# Display image (mid)
			ax2.set_title('Acquired Image')
			if log_im:
				ax2.imshow(np.log10(np.abs(self.img)), cmap='gray')
			else:
				ax2.imshow(np.abs(self.img), cmap='gray')
			
			fig.suptitle(title)

			# Display fft along readout (bot)
			fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
			ax3.set_title(f'FFT Along Readout')
			ax3.set_xlabel(r'$\frac{\omega}{\pi}$')
			ax3.set_ylabel('Phase Encode #')
			if log_ro:
				ax3.imshow(np.log10(np.abs(fft_readout)), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
			else:
				ax3.imshow(np.abs(fft_readout), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
			
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
			self.true_inds.append(true_ind)
			
			# Adjust the pseudo random sequence
			prnd_seq = np.roll(prnd_seq, -samples_added)
			
			# Device signal is then modulation * pilot tone * rnd sequence
			pt_sig_device = pt_amp * modulation[pe] * np.exp(2j*np.pi*freq*(n + samples_added) / self.fs_pt) * prnd_seq[:N_pt_ro]
			
			# The scanner receivess a resampled version of the original PT signal due to 
			# a mismatch in the sacnner BW and the PT device BW
			pt_sig_scanner = sig_utils.my_resample(pt_sig_device, n_ro, N_pt_ro)
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
		
		
		rng = np.random.RandomState(seed)

		if type == 'bern':
			self.prnd_seq = 2 * rng.randint(0,2,seq_len) - 1 + 1j * (2 * rng.randint(0,2,seq_len) - 1)
		elif type == 'norm':
			self.prnd_seq = rng.normal(0, 1, seq_len) + 1j * rng.normal(0, 1, seq_len)
		elif type == 'unif':
			self.prnd_seq = rng.uniform(-1, 1, seq_len) + 1j * rng.uniform(-1, 1, seq_len)
		elif type == 'pois':
			lambd = 0.01
			self.prnd_seq = rng.poisson(lambd, seq_len) + 1j * rng.poisson(lambd, seq_len)
		else:
			print('Invalid rnd sequence. L8r')
			quit()

		if mode == 'real':
			self.prnd_seq = np.real(self.prnd_seq)

		self.prnd_seq = np.sqrt(seq_len) * self.prnd_seq / np.linalg.norm(self.prnd_seq)		
			
		N_pt_ro = int(self.ksp.shape[1] * self.fs_pt / self.fs)
		if seq_len < N_pt_ro:
			self.prnd_seq = np.tile(self.prnd_seq, int(np.ceil(N_pt_ro / len(self.prnd_seq))))
			
	# Plots the standard deviation across each readout
	def get_ksp_std(self):
		return np.std(self.ksp, axis=1)