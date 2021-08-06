import numpy as np
import matplotlib.pyplot as plt
from math import ceil

class sig_utils:

	# My circular correlation function
	# Returns the max value and corelation array
	def my_cor(x, y):
		# Zero-pad if needed
		if len(x) > len(y):
			y = np.concatenate((y, np.zeros(len(x) - len(y))))
		elif len(y) > len(x):
			x = np.concatenate((x, np.zeros(len(y) - len(x))))
		
		# Length of signal
		N = len(x)

		# Compute correlation <CAN OPTIMIZE>
		cors = []
		for k in range(N):
			cors.append(np.dot(x, np.roll(y, k)))
		cors = np.array(cors)
		return np.max(cors), cors

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
	def __init__(self, tr=0, bwpp=0, fc=127.8e6):	
		# Timing parameters
		self.TR  = tr
		self.BWPP = bwpp

		# Frequencies
		self.fc = fc
		self.fs = None # Will be calculated soon

		# image and kspace of that image
		self.img = None
		self.ksp = None

		# PRND sequence for pilot tone
		self.prnd_seq = None

		# True modulation
		self.true_motion = []
	
	# Load an MR image (image space)
	def load_image(self, img):
		self.img = img
		self.ksp = sig_utils.fft2c(img)
		self.fs = self.ksp.shape[1] * self.BWPP
	
	# Load a K-space image
	def load_kspace(self, ksp):
		self.img = sig_utils.ifft2c(ksp)
		self.ksp = ksp
		self.fs = self.ksp.shape[1] * self.BWPP
	
	# Display image and kspace
	def MRshow(self, drng=1e-6, log_ksp=True, log_ro=False, log_im=False, title=''):
		# Check that we have images
		if self.img is None or self.ksp is None:
			print('Load image first')
			return
		
		if self.prnd_seq is not None:
			# Generate subplots
			fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))

			# Display kspace (top left)
			ax1.set_title('K-Space')
			if log_ksp:
				ax1.imshow(np.log(np.abs(self.ksp) + drng), cmap='gray')
			else:
				ax1.imshow(np.abs(self.ksp), cmap='gray')
			
			# Display image (top right)
			ax2.set_title('Acquired Image')
			if log_im:
				ax2.imshow(np.log10(np.abs(self.img)), cmap='gray')
			else:
				ax2.imshow(np.abs(self.img), cmap='gray')
			
			fig.suptitle(title)

			# Display fft along readout (bottom left)
			fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
			ax3.set_title(f'FFT Along Readout')
			ax3.set_xlabel(r'$\frac{\omega}{\pi}$')
			ax3.set_ylabel('Phase Encode #')
			if log_ro:
				ax3.imshow(np.log10(np.abs(fft_readout)), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
			else:
				ax3.imshow(np.abs(fft_readout), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
			
			# Display correlated fft along raeadout (bottom right)
			ax4.set_title('Auto-Correlation of RND Sequence')
			ax4.set_xlabel('Correlation Index')
			ax4.set_ylabel('Auto-Correlation')
			_, auto = sig_utils.my_cor(self.prnd_seq, self.prnd_seq)
			inds = np.arange(-len(auto)//2, len(auto)//2) 
			ax4.plot(inds, np.fft.fftshift(auto))
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
	def add_PT(self, freq, tr_uncert=0, modulation=None):
		# Number of phase encodes and readouts
		n_pe, n_ro = self.ksp.shape

		# Keep sequence of ones if no random sequence is selected
		if self.prnd_seq is None:
			prnd_seq = np.ones(self.ksp.shape[1])
		else:
			prnd_seq = self.prnd_seq
		
		# If modulation, just make a function that returns 1
		if modulation is None:
			modulation = lambda x : 1

		N = len(prnd_seq)
		# Add pilot tone to kspace data 
		for pe in range(n_pe):
			phase_accrued = (pe * self.TR)
			if tr_uncert != 0:
				phase_accrued *= 1 + (((2 * np.random.rand()) - 1) * tr_uncert)
			samples_accrued = int(phase_accrued * self.fs)
			for ro in range(n_ro):
				t = phase_accrued + ro / self.fs
				if ro == 0:
					self.true_motion.append(modulation(t))
				self.ksp[pe, ro] += modulation(t) * np.exp(2j*np.pi*freq*t) * prnd_seq[(samples_accrued + ro) % N]
		
		# Recalculate image
		self.img = sig_utils.ifft2c(self.ksp)

		# Calculate pilot tone location
		val = freq / self.fs
		beta = np.round(val) - val
		b = np.round(n_ro/2 + beta * n_ro)

		val = freq * self.TR
		alpha = np.round(val) - val
		a = np.round(n_pe/2 + alpha * n_pe)

		return a, b

	# Extracts Motion signal from Pilot Tone
	def motion_extract(self, fpt):
		# Standard Pilot Tone procedure
		if self.prnd_seq is None:
			amps = []
			pt_sig = np.exp(2j*np.pi*fpt*np.arange(self.ksp.shape[1])/self.fs)
			for ro in self.ksp:
				amps.append(np.abs(np.vdot(pt_sig, ro)))
			return np.array(amps) / self.ksp.shape[1]
		# Spread Spectrum procedure
		else:
			amps = []
			for i, ro in enumerate(np.real(self.ksp)):
				max, cors = sig_utils.my_cor(ro, self.prnd_seq)
				amps.append(max / self.ksp.shape[1])
			amps = np.array(amps)
			return amps
	
	# Generates a single sequence that will repeat itself
	def prnd_seq_gen(self, p=None, start_state=None, seq_len=None):
		if seq_len is None:
			seq_len = self.ksp.shape[1]
		prnd_seq = []
		# LFSR
		if p is None and start_state is not None:
			self.prnd_seq = sig_utils.lfsr_gen(seq_len, lfsr=start_state)
		# Binomial
		elif p is not None and start_state is None:
			self.prnd_seq = 2 * np.random.binomial(1, p, seq_len) - 1
		# Pure orthogonal hadmard codes
		else:
			self.prnd_seq = 2 * np.random.randint(0, 2, seq_len) - 1
			
	# Plots the standard deviation across each readout
	def get_ksp_std(self):
		return np.std(self.ksp, axis=1)