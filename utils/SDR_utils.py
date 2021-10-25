import matplotlib.pyplot as plt
import numpy as np
import struct
import os


def plot_fft(time_sig, fs, fc=0, n_fft=4096):
	fft = np.fft.fftshift(np.fft.fft(time_sig, n=n_fft))
	fft_axis = np.linspace(fc + -fs/2,fc + fs/2, n_fft)
	plt.figure()
	plt.plot(fft_axis, np.abs(fft))

class RTL_utils:

	def rtl_read(self, freq, rate, num_samples, gain, use_sdr=True, filename='rtl_captures/read.dat'):
		if use_sdr:
			# Call `rtl_sdr` application
			cmd =  'rtl_sdr ' + filename
			cmd += ' -f ' + str(freq)
			cmd += ' -s ' + str(rate)
			cmd += ' -n ' + str(num_samples)
			cmd += ' -g ' + str(gain)
			os.system(cmd)

		# Open SDR data file
		with open(filename, 'rb') as f:
			bytes_read = list(f.read())

		# read IQ
		I = np.array(bytes_read[::2]) - 127.5
		Q = np.array(bytes_read[1::2]) - 127.5
		time_data = I + 1j * Q

		# return time data
		return time_data
	
	def view_spectrum(self, iq_sig, freq, rate, n_avg=1, n_fft=4096):
		# Break reshape as matrix to optimize many np.fft operations
		iq_siq = iq_sig[:n_avg * (len(iq_sig) // n_avg)]
		iq_siq = iq_sig.reshape((n_avg, -1))

		# Take N_AVG FFTs 
		fft_mag_avg = np.mean(np.abs(np.fft.fftshift(np.fft.fft(iq_siq, n=n_fft, axis=1), axes=1)) ** 2, axis=0)
		fft_axis = np.linspace(freq - rate/2, freq + rate/2, n_fft) / 1e6
			
		# Plot decibel scale
		plt.figure()
		plt.plot(fft_axis, fft_mag_avg)
		plt.xlabel('MHz')
		plt.show()

class UHD_utils:

	def __init__(self, uhd_dir):
		UHD_utils.UHD_DIRECTORY = uhd_dir

		# Important directories
		UHD_utils.PY_DIR = os.getcwd().replace('\\', '/')
		UHD_utils.BIN_DIRECTORY = UHD_utils.UHD_DIRECTORY + '/bin'
		UHD_utils.EXEC_DIRECTORY = UHD_utils.UHD_DIRECTORY + '/lib/uhd/examples'

		# Important files
		UHD_utils.READ_SAMPLES = UHD_utils.EXEC_DIRECTORY + '/rx_samples_to_file'
		UHD_utils.WRITE_SAMPLES = UHD_utils.EXEC_DIRECTORY + '/tx_samples_from_file'

		# Change to BIN Directory since it contains the .dll file
		os.chdir(UHD_utils.BIN_DIRECTORY)

	"""
	Uses the SDR in transmit mode.
	Args:
		iq_sig - array of complex numbers to transmit, <complex nparray>
		freq - center frequency (Hz), <float>
		rate - DAC rate (samples/sec), <float>
		gain - gain of the SDR (dB), <float>
		bw - bandwidth of analog filter <float>
		file - Filename to write samples from, <string>
		repeat - do you want to repeat the signal infinitely?, <bool>
	"""
	def uhd_write(self, iq_sig, freq, rate=5e6, gain=10, bw=None, file='uhd_iq/write.dat', repeat=False):

		# File to write from
		file = UHD_utils.PY_DIR + '/' + file

		iq_sig.tofile(file)
		# # Convert iq values to byte array
		# b = bytes()
		# for i, comp in enumerate(iq_sig):
		# 	b += struct.pack("<f", float(np.real(comp)))
		# 	b += struct.pack("<f", float(np.imag(comp)))

		# # write to file
		# with open(file, 'wb') as f:
		# 	f.write(b)
		
		# Construct bash command to execute
		cmd = '"' + UHD_utils.WRITE_SAMPLES + '"'
		cmd += ' --freq ' + str(freq)
		cmd += ' --rate ' + str(rate)
		cmd += ' --type float'
		cmd += ' --gain ' + str(gain)
		if bw is not None:
			cmd += ' --bw ' + str(bw)
		cmd += ' --file ' + file
		if repeat:
			cmd += ' --repeat'
		
		# Now we execute the command and wait for temp_file to be populated
		os.system(cmd)

	"""
	Uses the SDR in receive mode.
	Args:
		freq - center frequency (Hz), <float>
		rate - DAC rate (samples/sec), <float>
		gain - gain of the SDR (dB), <float>
		duration - How long are we reading for (sec)?, <float>
		file - Filename to read samples into, <string>
	Returns:
		iq_sig - array of complex numbers, <complex nparray>
	"""
	def uhd_read(self, freq, rate=5e6, gain=10, duration=1, file='uhd_iq/write.dat'):

		# file to read samples into
		file = UHD_utils.PY_DIR + '/' + file
		
		# Construct bash command to execute
		cmd = '"' + UHD_utils.READ_SAMPLES + '"'
		cmd += ' --freq ' + str(freq)
		cmd += ' --rate ' + str(rate)
		cmd += ' --type float'
		cmd += ' --gain ' + str(gain)
		cmd += ' --duration ' + str(duration)
		cmd += ' --file ' + file

		# Now we execute the command and wait for temp_file to be populated
		os.system(cmd)

		# Read the samples into an IQ array:
		with open(file, 'rb') as f:
			bytes_read = f.read()
		
		# read IQ as floats
		I = []
		Q = []
		for i in range(0, len(bytes_read), 8):
			I.append(struct.unpack('<f', bytes_read[i:i+4]))
			Q.append(struct.unpack('<f', bytes_read[i+4:i+8]))

		iq_sig = np.array(I) + 1j * np.array(Q)
		return iq_sig