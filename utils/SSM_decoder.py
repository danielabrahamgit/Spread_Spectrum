import sys
import os
sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import numpy as np
import sys

from utils.sig_utils import sig_utils
from numpy.fft import fft, fftshift, ifft
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
		self.rnd_inds = []

	def motion_estimate_iq(self, iq_sig, mode='RSSM', chop=20):
		N = len(iq_sig)
		iq_sig = iq_sig[:(N // chop) * chop]
		ksp = np.array([iq_sig[chop*i:chop*(i+1)] for i in range(len(iq_sig) // chop)])		
		self.ro_dir='LR'
			
		return self.motion_estimate_ksp(ksp, mode=mode)

	def motion_estimate_ksp(self, ksp, mode='RSSM'):		
		# Number of phase encodes and readout length
		if self.ro_dir == 'LR':
			npe, ro_len = ksp.shape
		else:
			ro_len, npe = ksp.shape

		# number of PT samples in a readout
		N = int(ro_len * SSM_decoder.PT_BW / self.mr_bw)

		# Motion estimate to return
		est = np.zeros(npe, dtype=np.complex128)

		# Standard Pilot Tone procedure
		if mode == 'standard':
			n_fft = ksp.shape[1]
			if self.ro_dir == 'LR':
				fft_ro = fftshift(np.fft.fft(ksp, axis=1, norm='forward'), axes=1)
				ind_pt = np.argmax(np.sum(np.abs(fft_ro) ** 2, axis=0))
				est = fft_ro[:,ind_pt]
			else:
				fft_ro = fftshift(np.fft.fft(ksp, axis=0, norm='forward'), axes=0)
				ind_pt = np.argmax(np.sum(np.abs(fft_ro) ** 2, axis=1))
				est = fft_ro[ind_pt, :]
		# Robust SSM procedure
		elif mode == 'RSSM':
			# Possible random sequences 
			if len(self.prnd_seq) < N:
				self.prnd_seq = np.tile(self.prnd_seq, int(np.ceil(N / len(self.prnd_seq))))

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
				
				demod_sig = sig_up * self.doppler_exp

				# Motion extraction via circular correlation
				cor = sig_utils.my_cor(self.prnd_seq, demod_sig)

				# if i % 50 == 0:
				# 	plt.plot(np.abs(cor))
				# 	plt.show()

				ind = np.argmax(np.abs(cor))
				self.rnd_inds.append(ind)
				rnd = np.roll(self.prnd_seq, -ind)[:N]
				est[i] = np.mean(demod_sig * rnd.conj())

		return est

	def motion_esimate_ksp_multi(self, ksp_frames, mode='RSSM'):
		# kx, ky, and frame number on third dimention
		assert len(ksp_frames.shape) == 3

		motion = None
		for i in range(ksp_frames.shape[2]):
			frame = ksp_frames[:,:,i]
			est_frame_i = self.motion_estimate_ksp(frame, mode=mode)
			if motion is None:
				motion = est_frame_i
			else:
				motion = np.concatenate((motion, est_frame_i))
		
		return motion

	def estimate_doppler(self, sig_up):	
		N = len(sig_up)	
		n_fft = len(self.prnd_seq)

		C = fft(self.prnd_seq, n=n_fft).conj()
		S = fft(sig_up, n=n_fft)

		shift_l = np.round(-self.doppler_range * n_fft / self.PT_BW).astype(int)
		shift_r = np.round( self.doppler_range * n_fft / self.PT_BW).astype(int)

		S_shifts = np.array([np.roll(S, -i) for i in range(shift_l, shift_r + 1)])

		mults = C * S_shifts
		xcorrs = ifft(mults, axis=1)
		m = np.argmax(np.abs(xcorrs))
		rnd_ind = m % n_fft
		rnd = np.roll(self.prnd_seq, rnd_ind)[:N]
		k_est = m // n_fft + shift_l

		omega_low = 2 * np.pi * (k_est - 1) / n_fft
		omega_high = 2 * np.pi * (k_est + 1) / n_fft
		omegas = np.linspace(omega_low, omega_high, 10000)
		
		exps = np.exp(-1j * np.outer(omegas, np.arange(N)))
		B = np.sum(exps * sig_up * rnd.conj(), axis=1).flatten()
		self.doppler_omega = omegas[np.argmax(np.abs(B))]

		# self.doppler_omega = 2 * np.pi * (0.5) / self.PT_BW

		print(f'Doppler Estimate = {self.doppler_omega * self.PT_BW / (2e3 * np.pi)} (kHz)')
		self.doppler_exp = np.exp(-1j * self.doppler_omega * np.arange(N))

if __name__ == '__main__':
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		print('Expecting at kspace input file as argument')

	