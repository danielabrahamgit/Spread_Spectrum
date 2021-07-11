from PIL.Image import new
import numpy as np
import matplotlib.pyplot as plt

class sig_utils:
	Fs = 1e6
	def __init__(self, fs):
		sig_utils.Fs = fs

	# Generates random noise with specified type
	def gen_guassian_noise(sig_len, mean=0, std=0.1):
		return np.random.normal(mean, std, sig_len)

	# Generates random white noise 
	def gen_white_noise(sig_len, scale=1):
		return scale * (1 - 2 * np.random.random(sig_len))
	
	# Generates the signal and time data for a pure tone
	def gen_tone(f_tone, t_start, t_end, complex=False):
		# Generate time data
		t = np.arange(t_start, t_end, 1 / sig_utils.Fs)

		if complex:
			# Generate complex exponential tone
			tone = np.exp(2j * np.pi * f_tone * t)
		else:
			# Generate pure tone (cosine)	
			tone = np.cos(2 * np.pi * f_tone * t)

		# Return both
		return t, tone
	
	def plot_fft(sig, n_fft=None, one_sided=False):
		# Default to length of signal = fft number of points
		if n_fft is None:
			n_fft = len(sig)

		# First take fft (zero centered)
		sig_fft = np.abs(np.fft.fftshift(np.fft.fft(sig, n=n_fft)))

		# Generate freqency axis
		f = np.linspace(-sig_utils.Fs/2, sig_utils.Fs/2, len(sig_fft))

		# Plot data
		if one_sided:
			plt.plot(f[n_fft//2:], sig_fft[n_fft//2:])
		else:
			plt.plot(f, sig_fft)
		plt.show()
	
	def plot_sig(t, sig):
		plt.plot(t, sig)
		plt.show()


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
	def MRshow(self, drng=1e-6, log=True):
		# Check that we have images
		if self.img is None or self.ksp is None:
			print('Load image first')
			return

		# Generate subplots
		fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
		
		# Display kspace (left)
		ax1.set_title('K-Space')
		if log:
			ax1.imshow(np.log(np.abs(self.ksp) + drng), cmap='gray')
		else:
			ax1.imshow(np.abs(self.ksp), cmap='gray')

		# Display fft along raeadout (middle)
		fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
		ax2.set_title(f'FFT Along Readout')
		ax2.set_xlabel(r'$\frac{\omega}{\pi}$')
		ax2.imshow(np.abs(fft_readout), cmap='gray', extent=[-1, 1, self.ksp.shape[1], 0], aspect=2/self.ksp.shape[1])

		# Display image (right)
		ax3.set_title('Acquired Image')
		ax3.imshow(np.abs(self.img), cmap='gray')

		plt.show()

	# 2DFT and inverse 
	def fft2c(f):
		return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))
	def ifft2c(F):
		return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))
	
	# Adds a pure pilot tone to our kspace data
	def add_PT(self, freq, modulation=None):
		# Demodulate by center frequency
		new_freq = freq - self.fc
		
		# Use signal util to generate a pure tone
		sig_utils.Fs = self.fs
		n_pe, n_ro = self.ksp.shape
		_, sig = sig_utils.gen_tone(new_freq, 0, self.TR * n_pe, complex=True)
		sig *= 10

		# Add pilot tone to kspace data 
		sample_num = 0
		for pe in range(n_pe):
			# Add phase accumulated by waiting until the next readout
			sample_num += int(self.fs * (self.TR - (1 / self.BWPP)))
			for ro in range(n_ro):
				self.ksp[pe, ro] += sig[sample_num]
				sample_num += 1
		
		# Recalculate image
		self.img = MR_utils.ifft2c(self.ksp)