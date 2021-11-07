import sys
import os
from numpy.core.numeric import correlate

import scipy
sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import numpy as np
import sys

from utils.sig_utils import sig_utils
from numpy.fft import fft, fftshift, ifft
from scipy import signal


class SSM_decoder:

	def __init__(self, prnd_seq, ksp, mr_bw, pt_bw=250e3, pt_fc=0, doppler_range=None, ro_dir='LR'):
		# Save MR/PT bandwidth for future computation
		self.mr_bw = mr_bw
		self.pt_bw = pt_bw
		self.pt_fc = pt_fc

		# Pseudo random number sequence
		self.prnd_seq = prnd_seq

		# Doppler frequency uncertanty (in Hz)
		self.doppler_range = doppler_range
		self.doppler_omega = None
		self.doppler_exp = None

		# Readout direction
		self.ro_dir = ro_dir

		self.ksp = ksp
		self.corrected_ksp = np.zeros_like(ksp)

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
		N = int(ro_len * self.pt_bw / self.mr_bw)

		# Motion estimate to return
		est = np.zeros(npe, dtype=np.complex128)

		# kspace like matrix that will be populatede with PT contributions
		ksp_est = np.zeros_like(ksp)

		# Standard Pilot Tone procedure
		if mode == 'standard':
			n_fft = ksp.shape[1]
			if self.ro_dir == 'LR':
				fft_ro = fftshift(np.fft.fft(ksp, axis=1), axes=1)
				ind_pt = np.argmax(np.sum(np.abs(fft_ro) ** 2, axis=0))
				est = fft_ro[:,ind_pt]
			else:
				fft_ro = fftshift(np.fft.fft(ksp, axis=0, norm='forward'), axes=0)
				ind_pt = np.argmax(np.sum(np.abs(fft_ro) / np.linalg.norm(fft_ro,axis=0) ** 2, axis=1))
				est = fft_ro[ind_pt, :]
			
			# print(ind_pt)

			# plt.imshow(np.abs(fft_ro))
			# plt.show()
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

				ind = np.argmax(np.abs(cor))
				rnd = np.roll(self.prnd_seq, -ind)[:N]
				est[i] = np.sum(demod_sig * rnd.conj()) / N

				remove = sig_up - est[i] * rnd
				cor = sig_utils.my_cor(rnd, remove)

				ro_est = signal.resample_poly(sig_up -  est[i] * rnd, ro_len, N)

				if self.ro_dir == 'LR':
					ksp_est[i, :] = ro_est
				else:
					ksp_est[:, i] = ro_est

		return est, ksp_est

	def motion_estimate(self, mode='RSSM'):
		# kx, ky, and frame number on third dimention
		assert len(self.ksp.shape) >= 2
		ksp_est = np.zeros_like(self.ksp)
		motion = None
		if len(self.ksp.shape) == 3:
			ksp_frames = self.ksp
			for i in range(ksp_frames.shape[2]):
				frame = ksp_frames[:,:,i]
				est_frame_i, ksp_est_i = self.motion_estimate_ksp(frame, mode=mode)
				if motion is None:
					motion = est_frame_i
				else:
					motion = np.concatenate((motion, est_frame_i))
				ksp_est[:,:,i] = ksp_est_i
		else:
			motion, ksp_est = self.motion_estimate_ksp(self.ksp, mode=mode)
		
		return motion, ksp_est

	def estimate_doppler(self, sig_up):	
		N = len(sig_up)	
		n_fft = len(self.prnd_seq)

		C = fft(self.prnd_seq, n=n_fft).conj()
		S = fft(sig_up, n=n_fft)

		shift_l = np.round(-self.doppler_range * n_fft / self.pt_bw).astype(int)
		shift_r = np.round( self.doppler_range * n_fft / self.pt_bw).astype(int)

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

		# self.doppler_omega = 2 * np.pi * (0.0) / self.pt_bw

		print(f'Doppler Estimate = {self.doppler_omega * self.pt_bw / (2e3 * np.pi)} (kHz)')
		self.doppler_exp = np.exp(-1j * self.doppler_omega * np.arange(N))


	def peak_removal(self, ksp, est, thresh=20, deg=4):
		assert len(ksp.shape) == 2
		if self.ro_dir == 'LR':
			stds = np.std(ksp, axis=1).flatten()
		else:
			stds = np.std(ksp, axis=0).flatten()
		
		assert len(stds) == len(est)

		# plt.subplot(311)
		# plt.plot(stds)

		mid = len(est) // 2
		n = np.arange(len(est)) - mid
		
		# plt.subplot(312)
		# plt.plot(est)
		
		corrupt = np.where(stds > thresh)[0]
		not_corrupt = np.where(stds <= thresh)[0]
		coeff = np.polyfit(not_corrupt - mid, est[not_corrupt], deg)
		p = np.poly1d(coeff)
		est[corrupt] = p(corrupt - mid)

		# plt.subplot(313)
		# plt.plot(est)
		# plt.plot(p(n))
		# plt.show()




