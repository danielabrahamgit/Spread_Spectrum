import numpy as np

"""
Generates a pseudo random sequence of 1s and -1s 
Args:
	N - the number of points in the random seqence, <int>
	prnd_type - The type of sequence, <str>
Returns:
	prnd_seq - array of 1s and -1s, <np.array>
"""
def gen_prnd(N, prnd_type='bern'):
	
	# Bernoulli Random Sequence
	if prnd_type == 'bern':
		prnd_seq = 2 * np.random.randint(0, 2, N) - 1
	# Sad Random Sequence
	else:
		print('Enter a valid prnd sequence type')
		prnd_seq = None

	return prnd_seq