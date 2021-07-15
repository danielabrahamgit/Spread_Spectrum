import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
	
	# Load an MR image (image space)
	def load_image(self, img):
		self.img = img
		self.ksp = MR_utils.fft2c(img)
		self.fs = self.ksp.shape[1] * self.BWPP
	
	# Load a K-space image
	def load_kspace(self, ksp):
		self.img = MR_utils.ifft2c(ksp)
		self.ksp = ksp
		self.fs = self.ksp.shape[1] * self.BWPP
	
	# Display image and kspace
	def MRshow(self, drng=1e-6, log_ksp=True, log_ro=False, log_im=False, title=''):
		# Check that we have images
		if self.img is None or self.ksp is None:
			print('Load image first')
			return

		# Generate subplots
		fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 6))
		
		# Display kspace (left)
		ax1.set_title('K-Space')
		if log_ksp:
			ax1.imshow(np.log(np.abs(self.ksp) + drng), cmap='gray')
		else:
			ax1.imshow(np.abs(self.ksp), cmap='gray')

		# Display fft along raeadout (middle)
		fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
		ax2.set_title(f'FFT Along Readout')
		ax2.set_xlabel(r'$\frac{\omega}{\pi}$')
		if log_ro:
			ax2.imshow(np.log10(np.abs(fft_readout)), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
		else:
			ax2.imshow(np.abs(fft_readout), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])
		# Display image (right)
		ax3.set_title('Acquired Image')
		if log_im:
			ax3.imshow(np.log10(np.abs(self.img)), cmap='gray')
		else:
			ax3.imshow(np.abs(self.img), cmap='gray')
		
		fig.suptitle(title)
		plt.show()

	# 2DFT and inverse 
	def fft2c(f):
		return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))
	def ifft2c(F):
		return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))
	
	# Adds a pure pilot tone to our kspace data
	def add_PT(self, freq, tr_uncert=0, modulation=None, prnd_seq=None):
		# Number of phase encodes and readouts
		n_pe, n_ro = self.ksp.shape

		# Keep sequence of ones if no random sequence is selected
		if prnd_seq is None:
			prnd_seq = np.ones(n_ro * n_pe)

		# Add pilot tone to kspace data 
		for pe in range(n_pe):
			phase_accrued = (pe * self.TR)
			if tr_uncert != 0:
				phase_accrued *= 1 + (((2 * np.random.rand()) - 1) * tr_uncert)
			for ro in range(n_ro):
				t = phase_accrued + ro / self.fs
				if modulation is None:
					self.ksp[pe, ro] += 2700 * np.exp(2j*np.pi*freq*t) * prnd_seq[n_ro * pe + ro]
				else: 
					self.ksp[pe, ro] += modulation(t) * np.exp(2j*np.pi*freq*t) * prnd_seq[n_ro * pe + ro]

		# Recalculate image
		self.img = MR_utils.ifft2c(self.ksp)

		# Calculate pilot tone location
		val = freq / self.fs
		beta = np.round(val) - val
		b = np.round(n_ro/2 + beta * n_ro)

		val = freq * self.TR
		alpha = np.round(val) - val
		a = np.round(n_pe/2 + alpha * n_pe)

		return a, b

	# Extracts Motion signal from Pilot Tone
	def motion_extract(self, fpt=None, prnd_seq=None):
		# Correct kspace if needed
		if prnd_seq is not None:
			prnd_seq = np.resize(np.array(prnd_seq), self.ksp.shape)
			new_ksp = self.ksp * prnd_seq
		else:
			new_ksp = self.ksp

		# Plots the readout FFTs before and after correlation
		fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
		plt.subplot(2,1,1)
		plt.imshow(np.log10(np.abs(fft_readout)), cmap='gray')

		# First we take the K-Space FFT along the readout direction
		fft_readout = np.fft.fftshift(np.fft.fft(new_ksp, axis=1), axes=1)
		n_pe, n_ro = new_ksp.shape

		# Plots the readout FFTs before and after correlation
		plt.subplot(2,1,2)
		plt.imshow(np.log10(np.abs(fft_readout)), cmap='gray')
		plt.figure()

		if fpt == None:
			# Now we search for a strong vertical line corelation
			amax = np.argmax(np.sum(np.abs(fft_readout), axis=0))
		else:
			amax = int(np.round(n_ro * (fpt / self.fs + 0.5)))
		
		# # Now that we have the pilot tone frequency, we want to filter our kspace 
		# margin = 1e3
		# fpt = np.abs((amax/n_ro - 1/2) * self.fs)
		# filt = signal.firwin(129, [fpt - margin, fpt + margin], pass_zero=False, fs=self.fs)
		# new_ksp = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=1, arr=self.ksp)
		# fft_readout = np.fft.fftshift(np.fft.fft(new_ksp, axis=1), axes=1)

		# Finally exract the motion signal now that we have the index
		return fft_readout[:, amax]

	# Generates pseudo random sequence
	def prnd_seq_gen(self, p=None, start_state=None):
		prnd_seq = []
		# LFSR
		if p is None and start_state is not None:
			target_length = np.prod(self.ksp.shape)
			while len(prnd_seq) != target_length:
				next_bit = ((start_state >> 0) ^ (start_state >> 2) ^ (start_state >> 3) ^ (start_state >> 5)) & 1
				start_state = (next_bit << 16) | (start_state >> 1) 
				prnd_seq.append(1 - 2 * next_bit)
		# Binomial
		elif p is not None and start_state is None:
			flips = np.random.binomial(1, p, np.prod(self.ksp.shape))
			one = True
			for flip in flips:
				if flip:
					one = not one
				prnd_seq.append(1 - 2 * int(one == True))
		else:
			return None
		
		return prnd_seq