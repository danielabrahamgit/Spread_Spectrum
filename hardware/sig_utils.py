import numpy as np
import matplotlib.pyplot as plt

def view_spectrum(iq_sig, freq, rate, n_avg=1, n_fft=4096):
	# Break reshape as matrix to optimize many np.fft operations
	iq_siq = iq_sig[:n_avg * (len(iq_sig) // n_avg)]
	iq_siq = iq_sig.reshape((n_avg, -1))

	# Take N_AVG FFTs 
	fft_mag_avg = np.mean(np.abs(np.fft.fftshift(np.fft.fft(iq_siq, n=n_fft, axis=1), axes=1)), axis=0)
	fft_axis = np.linspace(freq - rate/2, freq + rate/2, n_fft) / 1e6
		
	# Plot decibel scale
	plt.figure()
	plt.plot(fft_axis, fft_mag_avg)
	plt.xlabel('MHz')
	plt.show()