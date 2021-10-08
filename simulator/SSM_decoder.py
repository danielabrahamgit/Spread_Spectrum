import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import sig_utils

class SSM_decoder:

	PT_BW = 500e3
	# relative to scanner center!
	PT_FC = 0

	def __init__(self, mr_bw, prnd_seq, mr_fc=127.7e6, pt_fc=None, pt_bw=None):
		# Save MR bandwidth for future computation
		self.mr_bw = mr_bw

		# Pseudo random number sequence
		self.prnd_seq = prnd_seq

		# Update PT parameters if needed
		if pt_fc is not None:
			SSM_decoder.PT_FC = pt_fc
		if pt_bw is not None:
			SSM_decoder.PT_BW = pt_bw

	def motion_estimate(self, ksp, mode='brute'):
		# Number of phase encodes and readout length
		npe, ro_len = ksp.shape
		N = int(ro_len * SSM_decoder.PT_BW / self.mr_bw)
		# Motion estimate 
		est = np.zeros(npe)
		inds = []

		if mode == 'brute':
			# First, we find all possible random sequences :(
			prnd_mat = np.array([np.roll(self.prnd_seq, -i)[:N] for i in range(len(self.prnd_seq))])

			# Now exhaustively search upon each readout :(
			for i, ro in enumerate(ksp):
				sig_up = sig_utils.my_resample(ro, N, ro_len)
				prnd_mults = sig_up * prnd_mat
				F = np.abs(np.fft.fft(prnd_mults, axis=1))
				est[i] = np.max(F) / F.shape[1]
		elif mode == 'ballpark':
			fc_expected = SSM_decoder.PT_FC
			fc_potential = fc_expected + np.linspace(-5,5,11)
			exps = np.exp(-2j * np.pi * np.outer(fc_potential, np.arange(N)))

			# Exchaustive search
			for i, ro in enumerate(ksp):
				sig_up = sig_utils.my_resample(ro, N, ro_len)
				sig_up *= np.exp(-2j * np.pi * 0 * np.arange(N))
				shift = np.argmax(np.abs(sig_utils.my_cor(sig_up, self.prnd_seq)))
				rnd = np.roll(self.prnd_seq, shift)[:len(sig_up)]
				sig_up *= rnd 
				est[i] = np.mean(np.abs(sig_up))
				# mults = sig_up * exps
				# Auto = sig_utils.auto_cor_mat(mults)
				# est[i] = np.max(np.abs(Auto))

		return sig_utils.normalize(est), inds


if __name__ == '__main__':
	if len(sys.argv) == 2:
		x = 0
	else:
		print('Expecting at kspace input file as argument')