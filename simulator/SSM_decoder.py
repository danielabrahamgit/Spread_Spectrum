import numpy as np
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
		# Motion estimate 
		est = np.zeros(npe)

		if mode == 'brute':
			# First, we find all possible random sequences :(
			N = int(ro_len * SSM_decoder.PT_BW / self.mr_bw)
			prnd_mat = np.array([np.roll(self.prnd_seq, -i)[:N] for i in range(len(self.prnd_seq))])

			# Now exhaustively search upon each readout :(
			for i, ro in enumerate(ksp):
				sig_up = sig_utils.my_resample(ro, N, ro_len)
				prnd_mults = sig_up * prnd_mat
				F = np.abs(np.fft.fft(prnd_mults, axis=1))
				est[i] = np.max(F) / F.shape[1]

		return sig_utils.normalize(est)


