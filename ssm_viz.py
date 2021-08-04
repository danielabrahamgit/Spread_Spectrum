import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fftshift, fft
from utils import MR_utils
from math import ceil

# Sampling Rate
Fs = 1e3
# Pilot Tone Frequency
Fpt = 0
# Number of FFT points
N_FFT = 2 ** 12
# Time axis
t = np.arange(0, 1, 1/Fs)
# Frequency axis
f = np.linspace(-Fs/2, Fs/2, N_FFT)
# Number of points
N = len(t)
# number of different SSM repeats
n_repeats = [1, 4, 10, 15, 25]

# PRND sequence
prnd_arr = np.array([np.repeat(1 - 2 * np.random.randint(0, 2, ceil(N / i)), i)[:N] for i in n_repeats] + [np.ones(N)])

# Pilot tone signal
x_pt = np.resize(np.exp(2j * np.pi * Fpt * t), (1, N))
x_pt_repeated = np.repeat(x_pt, prnd_arr.shape[0], axis=0)
# SSM signals
x_ssm = x_pt_repeated * prnd_arr

# FFT of SSM PTs
X_ssm = fftshift(fft(x_ssm, n=N_FFT, axis=1), axes=1)

# # Display both PT and SSM PT
# for i in range(X_ssm.shape[0]):
#     x = X_ssm[i, :]
	
#     if i < len(n_repeats):
#         l = f'Min Repeats = {n_repeats[i]}'
#     else:
#         l = 'No SSM'

#     plt.subplot(X_ssm.shape[0], 1, i + 1)
#     plt.plot(f, np.abs(x), label=l)
#     plt.legend()
# plt.show()

def hex_to_bits(hex_int):
	bits = []
	for i in range(16):
		bits.append(2 * (hex_int & 1) - 1)
		hex_int = hex_int >> 1
	return bits

def lfsr_gen(num_codes, code_len, lfsr1=0xACE1, lfsr2=0xf11e):
	nums = []
	for _ in range(num_codes):
		temp_bits = []
		for i in range(code_len // 16):
			bit1 = ((lfsr1 >> 0) ^ (lfsr1 >> 2) ^ (lfsr1 >> 3) ^ (lfsr1 >> 5)) & 1
			bit2 = ((lfsr2 >> 0) ^ (lfsr2 >> 2) ^ (lfsr2 >> 3) ^ (lfsr2 >> 5)) & 1
			lfsr1 = (lfsr1 >> 1) | (bit1 << 15)
			lfsr2 = (lfsr2 >> 1) | (bit2 << 15)
			temp_bits += hex_to_bits(lfsr2 ^ lfsr1)
		nums.append(np.array(temp_bits))

	return np.array(nums)

def hadamard(n):
	try:
		assert (n & (n-1) == 0) and n != 0
	except:
		print("'n' needs to be a power of 2")
	
	if n == 1:
		return 1
	
	h_half = hadamard(n//2)
	h_top = np.hstack((h_half, h_half))
	h_bot = np.hstack((h_half, -h_half))

	return np.vstack((h_top, h_bot))


num_codes = 128
code_len = 128
seq = lfsr_gen(num_codes, code_len, lfsr1=0xf11e, lfsr2=0xace1)
h = h_mat(2 ** 7)
print(h.shape)
print(h)
plt.imshow(h.T @ h)
plt.show()


