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

		# PRND sequence for pilot tone
		self.prnd_seq = None

		# True modulation
		self.true_motion = []
	
	# Load an MR image (image space)
	def load_image(self, img):
		self.img = img
		# n = 2
		# zp = np.zeros((img.shape[0], img.shape[1] * n))
		# diff = img.shape[1] * (n - 1)
		# zp[:, diff//2:-diff//2] = img
		# self.ksp = MR_utils.fft2c(zp)
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
		if self.prnd_seq is not None:
			if len(self.prnd_seq.shape) == 1:
				rows = []
				for ro in self.ksp:
					rows.append(np.correlate(ro, self.prnd_seq, mode='same'))
				new_ksp = np.array(rows)
			else:
				new_ksp = self.prnd_seq @ self.ksp.T
		else:
			new_ksp = self.ksp
		ax4.set_title(f'Log Correlation of Each Readout With PRND Seq')
		ax4.set_xlabel('Correlation Index')
		ax4.set_ylabel('Phase Encode #')
		ax4.imshow(np.log(np.abs(new_ksp)), cmap='gray', aspect=len(self.prnd_seq)/self.ksp.shape[1])
		plt.show()

	# 2DFT and inverse 
	def fft2c(f, shp=None):
		if shp == None:
			shp = f.shape
		return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f), s=shp))
	def ifft2c(F, shp=None):
		if shp == None:
			shp = F.shape
		return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F), s=shp))
	
	# Adds a pure pilot tone to our kspace data
	def add_PT(self, freq, tr_uncert=0, modulation=None):
		# Number of phase encodes and readouts
		n_pe, n_ro = self.ksp.shape

		# Determine what type of random sequence we are using
		if len(self.prnd_seq.shape) > 1:
			matrix = True
		else:
			matrix = False

		# Keep sequence of ones if no random sequence is selected
		if self.prnd_seq is None:
			if matrix:
				self.prnd_seq = np.ones(self.ksp.shape)
			else:
				self.prnd_seq = np.ones(self.ksp.shape[1])
		
		# If modulation, just make a function that returns 1
		if modulation is None:
			modulation = lambda x : 1

		self.ksp_og = self.ksp.copy()
		self.prnd_mat = np.zeros(self.ksp.shape)
		N = len(self.prnd_seq)
		# Add pilot tone to kspace data 
		for pe in range(n_pe):
			phase_accrued = (pe * self.TR)
			if tr_uncert != 0:
				phase_accrued *= 1 + (((2 * np.random.rand()) - 1) * tr_uncert)
			samples_accrued = int(phase_accrued * self.fs)
			for ro in range(n_ro):
				t = phase_accrued + ro / self.fs
				self.prnd_mat[pe,ro] = self.prnd_seq[(samples_accrued + ro) % N]
				if ro == 0:
					self.true_motion.append(modulation(t))
				if matrix:
					self.ksp[pe, ro] += modulation(t) * np.exp(2j*np.pi*freq*t) * self.prnd_seq[pe, ro]
				else:
					self.ksp[pe, ro] += modulation(t) * np.exp(2j*np.pi*freq*t) * self.prnd_seq[(samples_accrued + ro) % N]
		plt.figure()
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
	def motion_extract(self, fpt=None):
		
		if self.prnd_seq is None:
			# First we take the K-Space FFT along the readout direction
			fft_readout = np.fft.fftshift(np.fft.fft(self.ksp, axis=1), axes=1)
			n_pe, n_ro = self.ksp.shape

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

			# plt.plot(np.real(new_ksp[27, :]))
			# plt.figure()
			
			# Finally exract the motion signal now that we have the index
			motion_sig = fft_readout[:, amax]
			# self.ksp -= np.resize(np.repeat(motion_sig, self.ksp.shape[1]), self.ksp.shape) * self.prnd_seq
			# self.img = MR_utils.ifft2c(self.ksp)
			return motion_sig
		# Repeating single sequence
		elif len(self.prnd_seq.shape) == 1:
			amps = []
			for i, ro in enumerate(np.real(self.ksp)):
				amps.append(np.max(np.correlate(ro, self.prnd_seq, mode='same')))
			return np.array(amps) / self.ksp.shape[1]
		# Matrix with many random sequences
		else:
			amps = []
			for i, ro in enumerate(self.ksp):
				std = np.std(ro)
				prnd_seq_ro = self.prnd_seq[i, :]
				raw_inner_aprox = 0
				# if std > 100:
				# 	corr_ish = np.sum(np.conj(ro) * np.delete(self.prnd_seq, i, 0), axis=1)
				# 	raw_inner_aprox = np.mean(corr_ish)
				amps.append(np.vdot(ro, 1 / prnd_seq_ro) - raw_inner_aprox)
			return np.array(amps) / self.ksp.shape[1]
	
	# Generates purely orthogonal codes
	def hadamard(n):
		try:
			assert (n & (n-1) == 0) and n != 0
		except:
			print("'n' needs to be a power of 2")
		
		if n == 1:
			return 1
		
		h_half = MR_utils.hadamard(n//2)
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
	def lfsr_gen(self, lfsr=0xACE1):
		nums = []
		for _ in range(self.ksp.shape[0]):
			temp_bits = []
			for i in range(self.ksp.shape[1] // 16):
				bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1
				lfsr = (lfsr >> 1) | (bit << 15)
				temp_bits += MR_utils.hex_to_bits(lfsr)
			nums.append(np.array(temp_bits))
		return np.array(nums)

	# Generates pseudo random sequence
	def prnd_mat_gen(self, p=None, start_state=None):
		prnd_seq = []
		# LFSR
		if p is None and start_state is not None:
			prnd_seq = self.lfsr_gen(lfsr=start_state)[0,:]
		# Binomial
		elif p is not None and start_state is None:
			flips = np.random.binomial(1, p, np.prod(self.ksp.shape))
			one = True
			for flip in flips:
				if flip:
					one = not one
				prnd_seq.append(1 - 2 * int(one == True))
			prnd_seq = np.resize(np.array(prnd_seq), self.ksp.shape)
		# Pure orthogonal hadmard codes
		else:
			prnd_seq = MR_utils.hadamard(self.ksp.shape[1])
		
		self.prnd_seq = prnd_seq
	
	# Generates a single sequence that will repeat itself
	def prnd_seq_gen(self, p=None, start_state=None, seq_len=None):
		if seq_len is None:
			seq_len = self.ksp.shape[1]
		prnd_seq = []
		# LFSR
		if p is None and start_state is not None:
			self.prnd_seq = self.lfsr_gen(lfsr=start_state).flatten()[:seq_len]
		# Binomial
		elif p is not None and start_state is None:
			flips = np.random.binomial(1, p, seq_len)
			one = True
			for flip in flips:
				if flip:
					one = not one
				prnd_seq.append(1 - 2 * int(one == True))
			self.prnd_seq = np.array(prnd_seq)
		# Pure orthogonal hadmard codes
		else:
			self.prnd_seq = 2 * np.random.randint(0, 2, seq_len) - 1
			

	# Plots the standard deviation across each readout
	def get_ksp_std(self):
		return np.std(self.ksp, axis=1)