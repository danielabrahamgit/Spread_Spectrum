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
	def gen_tone(f_tone, t_start, t_end):
		# Generate time data
		t = np.arange(t_start, t_end, 1 / sig_utils.Fs)

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