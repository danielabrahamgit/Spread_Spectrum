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

	def __init__(self, mr_bw, prnd_seq, pt_fc=0, doppler_range=None, pt_bw=None, ro_dir='LR'):
		# Save MR bandwidth for future computation
		self.mr_bw = mr_bw

		# Pseudo random number sequence
		self.prnd_seq = prnd_seq

		# Doppler frequency uncertanty (in Hz)
		self.doppler_range = doppler_range
		self.doppler_omega = None
		self.doppler_exp = None

		# Readout direction
		self.ro_dir = ro_dir

		# Update PT parameters if needed
		if pt_fc is not None:
			SSM_decoder.PT_FC = pt_fc
		if pt_bw is not None:
			SSM_decoder.PT_BW = pt_bw

		self.prnd_mat = None

	def motion_estimate_iq(self, iq, mode='RSSM', normalize=True, chop=20):
		N = len(iq)
		iq = iq[:(N // chop) * chop]
		ksp = np.array([iq[chop*i:chop*(i+1)] for i in range(len(iq) // chop)])		
			
		return self.motion_estimate_ksp(ksp, mode=mode, normalize=normalize)

	def motion_estimate_ksp(self, ksp, mode='RSSM', normalize=True):		
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
			n_fft = ksp.shape[1]
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
				sig_up = signal.resample_poly(ro, N, ro_len)
				
				# Center Frequency estimation
				if self.doppler_omega is None:
					self.estimate_doppler(sig_up)

				# Motion extraction via circular correlation
				cor = sig_utils.my_cor(self.prnd_seq, sig_up * self.doppler_exp)
				
				# plt.plot(np.abs(cor))
				# plt.show()

				rnd = np.roll(self.prnd_seq, -np.argmax(np.abs(cor)))[:N]
				
				# Update estimate
				# est[i] = np.linalg.norm(sig_up)
				est[i] = np.abs(np.sum(rnd * sig_up * self.doppler_exp))
				# est[i] = np.max(np.abs(cor))

		# Normalize the motion estimate if needed
		if normalize:
			return sig_utils.normalize(est)
		else:
			return est

	def motion_esimate_ksp_multi(self, ksp_frames, mode='RSSM', normalize=True):
		# kx, ky, and frame number on third dimention
		assert len(ksp_frames.shape) == 3

		motion = None
		for i in range(ksp_frames.shape[2]):
			frame = ksp_frames[:,:,i]
			est_frame_i = self.motion_estimate_ksp(frame, mode=mode, normalize=False)
			if motion is None:
				motion = est_frame_i
			else:
				motion = np.concatenate((motion, est_frame_i))
		
		if normalize:
			return sig_utils.normalize(motion)
		else:
			return motion

	def estimate_doppler(self, sig_up):		
		N = len(sig_up)
		mults = sig_up * self.prnd_mat.conj()
		n_fft = 2 ** 11
		F = np.abs(fft(mults, axis=1, n=n_fft))
		exp_ind = np.argmax(F ** 2) % n_fft
		ind = np.argmax(F ** 2) // n_fft
		self.doppler_omega = (2 * np.pi * exp_ind / n_fft)
		if self.doppler_omega > np.pi:
			self.doppler_omega -= 2 * np.pi
		# self.doppler_omega = 2 * np.pi * (-5e3) / self.PT_BW
		print(f'Doppler Estimate = {self.doppler_omega * self.PT_BW / (2e3 * np.pi)} (kHz)')
		self.doppler_exp = np.exp(-1j * self.doppler_omega * np.arange(N))



if __name__ == '__main__':
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		print('Expecting at kspace input file as argument')

	