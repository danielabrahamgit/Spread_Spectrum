from matplotlib import use
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
		arg - serial number of USRP device
		clk - external clock?
	"""
	def uhd_write(self, iq_sig, freq, rate=5e6, gain=10, bw=None, file='uhd_iq/write.dat', repeat=False, arg=None, clk=False):

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
		if arg is not None:
			cmd += ' --args serial=' + arg
		if clk:
			cmd += ' --ref external'

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
		arg - Serial number of the USRP
		use_sdr - read from file or get new data with SDR?
		clk - External clock?
	Returns:
		iq_sig - array of complex numbers, <complex nparray>
	"""
	def uhd_read(self, freq, rate=5e6, gain=10, duration=1, file='uhd_iq/read.dat', arg=None, use_sdr=True, clk=False):

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
		if arg is not None:
			cmd += ' --args serial=' + arg
		if clk:
			cmd += ' --ref external'

		if use_sdr:
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
		return iq_sig.flatten()

	"""
		Helper function to store sequence parameters into a file
	"""
	def save_sequence_params(self, seq_id, prnd_seq_len, prnd_type, prnd_mode, prnd_seed, center_freq, tx_rate, tx_gain):
		seq_data = {}
		seq_data['prnd_seq_len'] = prnd_seq_len
		seq_data['prnd_type'] = prnd_type
		seq_data['prnd_mode'] = prnd_mode
		seq_data['prnd_seed'] = prnd_seed
		seq_data['center_freq'] = center_freq
		seq_data['tx_rate'] = tx_rate
		seq_data['tx_gain'] = tx_gain

		s = ''
		for key in seq_data.keys():
			s += key + ': '
			s += str(seq_data[key])
			s += '\n\n'
		with open(self.PY_DIR +  seq_id + '.txt', 'w') as f:
			f.write(s)
