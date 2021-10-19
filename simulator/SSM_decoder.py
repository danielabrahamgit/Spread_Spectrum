import sys
import os
sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import numpy as np
import sys

from utils.sig_utils import sig_utils
from numpy.fft import fft, fftshift
from scipy import signal

class SSM_decoder:

	# Pilot tone Bandwidth
	PT_BW = 500e3
	# Pilot tone cetner frequency relative to scanner center!
	PT_FC = 0

	def __init__(self, mr_bw, prnd_seq, pt_fc=0, pt_fc_uncert=0, pt_bw=None, ro_dir='LR'):
		# Save MR bandwidth for future computation
		self.mr_bw = mr_bw

		# Pseudo random number sequence
		self.prnd_seq = prnd_seq

		# Readout direction
		self.ro_dir = ro_dir

		# Update PT parameters if needed
		if pt_fc is not None:
			SSM_decoder.PT_FC = pt_fc
		if pt_bw is not None:
			SSM_decoder.PT_BW = pt_bw

		self.prnd_mat = None

		self.omega = None
		self.exp = None

	def motion_estimate(self, ksp, mode='brute', normalize=True):
		# Number of phase encodes and readout length
		if self.ro_dir == 'LR':
			npe, ro_len = ksp.shape
		else:
			ro_len, npe = ksp.shape

		# number of PT samples in a readout
		N = int(ro_len * SSM_decoder.PT_BW / self.mr_bw)
		
		# Motion estimate to return
		est = np.zeros(npe)

		# Standard Pilot Tone procedure
		if mode == 'standard':
			if self.ro_dir == 'LR':
				fft_ro = fftshift(np.fft.fft(ksp, axis=1), axes=1)
				ind_pt = np.argmax(np.sum(np.abs(fft_ro) ** 2, axis=0))
				est = np.abs(fft_ro[:,ind_pt])
			else:
				fft_ro = fftshift(np.fft.fft(ksp, axis=0), axes=0)
				ind_pt = np.argmax(np.sum(np.abs(fft_ro) ** 2, axis=1))
				est = np.abs(fft_ro[ind_pt, :])
		# Robust SSM procedure
		elif mode == 'RSSM':
			# Possible random sequences 
			if len(self.prnd_seq) < N:
				self.prnd_seq = np.tile(self.prnd_seq, int(np.ceil(N / len(self.prnd_seq))))
			if self.prnd_mat is None:
				self.prnd_mat = np.array([np.roll(self.prnd_seq, -i)[:N] for i in range(len(self.prnd_seq))])

			# Go through each readout
			for i in range(npe):
				if self.ro_dir == 'LR':
					ro = ksp[i, :]
				else:
					ro = ksp[:, i]

				# Upsample readout to PT sample rate
				sig_up = sig_utils.my_resample(ro, N, ro_len)
				
				# Center Frequency estimation
				if self.omega is None:
					mults = sig_up * self.prnd_mat.conj()
					n_fft = 2 ** 14
					F = np.abs(fft(mults, axis=1, n=n_fft))
					exp_ind = np.argmax(F) % n_fft
					self.omega = (2 * np.pi * exp_ind / n_fft)
					if self.omega > np.pi:
						self.omega -= 2 * np.pi
					# fc_error = 127700000 - 127698331
					# self.omega = -fc_error * 2 * np.pi / self.PT_BW
					print(self.omega * self.PT_BW / (2 * np.pi))
					self.exp = np.exp(-1j * self.omega * np.arange(N))

				# Motion extraction via circular correlation
				cor = sig_utils.my_cor(self.prnd_seq, sig_up * self.exp)
				
				# Update estimate
				est[i] = np.max(np.abs(cor))

		# Normalize the motion estimate if needed
		if normalize:
			return sig_utils.normalize(est)
		else:
			return est


if __name__ == '__main__':
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		print('Expecting at kspace input file as argument')

	