import numpy as np
from math import ceil
from scipy import signal
from numpy.fft import fft, fftshift


class sig_utils:
	
	# PRND Sequence Generation
	def prnd_gen(seq_len, type='bern', mode='real', seed=10):
		rng = np.random.RandomState(seed)

		if type == 'bern':
			prnd_seq = 2 * rng.randint(0,2,seq_len) - 1 + 1j * (2 * rng.randint(0,2,seq_len) - 1)
		elif type == 'norm':
			prnd_seq = rng.normal(0, 1, seq_len) + 1j * rng.normal(0, 1, seq_len)
		elif type == 'unif':
			prnd_seq = rng.uniform(-1, 1, seq_len) + 1j * rng.uniform(-1, 1, seq_len)
		elif type == 'pois':
			lambd = 0.01
			prnd_seq = rng.poisson(lambd, seq_len) + 1j * rng.poisson(lambd, seq_len)
		else:
			print('Invalid rnd sequence. L8r')
			quit()

		if mode == 'real':
			prnd_seq = np.real(prnd_seq)

		prnd_seq = np.sqrt(seq_len) * prnd_seq / np.linalg.norm(prnd_seq)

		return prnd_seq


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
		ntaps = min(N * up, ntaps)
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

