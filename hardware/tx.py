import matplotlib.pyplot as plt
import numpy as np
import struct
import os
from numpy.fft import fft, fftshift, ifft
from sig_utils import view_spectrum

# ------------- INSERT PATH TO THE UHD FOLDER HERE -------------
UHD_DIRECTORY = 'C:/Program Files (x86)/UHD'
# --------------------------------------------------------------

# Important directories
PY_DIR = os.getcwd().replace('\\', '/')
BIN_DIRECTORY = UHD_DIRECTORY + '/bin'
EXEC_DIRECTORY = UHD_DIRECTORY + '/lib/uhd/examples'

# Important files
READ_SAMPLES = EXEC_DIRECTORY + '/rx_samples_to_file'
WRITE_SAMPLES = EXEC_DIRECTORY + '/tx_samples_from_file'

# Change to BIN Directory since it contains the .dll file
os.chdir(BIN_DIRECTORY)

def sdr_write(iq_sig, freq, rate=5e6, gain=10, bw=None, file=None, repeat=False):
# Uses the SDR in transmit mode.
# Args:
# 	iq_sig - array of complex numbers to transmit, <complex nparray>
# 	freq - center frequency (Hz), <float>
# 	rate - DAC rate (samples/sec), <float>
# 	gain - gain of the SDR (dB), <float>
# 	bw - bandwidth of analog filter <float>
# 	file - Filename to write samples from, <string>
# 	repeat - do you want to repeat the signal infinitely?, <bool>
	
	if bw is None:
		bw = rate

	# File to write from
	if file is None:
		file = PY_DIR +  '/write.dat'

		# Convert iq values to byte array
		b = bytes()
		for i, comp in enumerate(iq_sig):
			b += struct.pack("<f", np.real(comp))
			b += struct.pack("<f", np.imag(comp))

		# write to file
		with open(file, 'wb') as f:
			f.write(b)
	
	# Construct bash command to execute
	cmd = '"' + WRITE_SAMPLES + '"'
	cmd += ' --freq ' + str(freq)
	cmd += ' --rate ' + str(rate)
	cmd += ' --type float'
	cmd += ' --gain ' + str(gain)
	cmd += ' --bw ' + str(bw)
	cmd += ' --file ' + file
	if repeat:
		cmd += ' --repeat'
	
	# Now we execute the command and wait for temp_file to be populated
	os.system(cmd)


def sdr_read(freq, rate=5e6, gain=10, duration=1, file=None):
# Uses the SDR in receive mode.
# Args:
# 	freq - center frequency (Hz), <float>
# 	rate - DAC rate (samples/sec), <float>
# 	gain - gain of the SDR (dB), <float>
# 	duration - How long are we reading for (sec)?, <float>
# 	file - Filename to read samples into, <string>
# Returns:
# 	iq_sig - array of complex numbers, <complex nparray>

	# file to read samples into
	if file is None:
		file = PY_DIR +  '/read.dat'
	
	# Construct bash command to execute
	cmd = '"' + READ_SAMPLES + '"'
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

freq = 677.887e6
rate = 1e6
duration = 0.01
# iq_sig = sdr_read(freq=freq, rate=rate, gain=20, duration=duration)
M=1
N = int(((duration * rate)//M) * M)
n = np.arange(N)
f = 0
r = 1 - 2*np.random.randint(0,2,N // M)
r = np.repeat(r, M)
iq_sig = np.exp(2j * np.pi * f * n / rate) * r
sdr_write(iq_sig, freq, rate, 30, bw=0.2e6, repeat=True)
# Show fft
n_avg = 10
# view_spectrum(iq_sig, freq, rate, n_avg)

