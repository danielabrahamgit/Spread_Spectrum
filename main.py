import numpy as np
import matplotlib.pyplot as plt
from utils import sig_utils, MR_utils
from PIL import Image

# Load MR image
im = np.array(Image.open('images/brain.png'))
# ksp = np.load('kspace/500_30sec.npz')['ksp'][:,:,0,3].T
ksp = np.random.normal(0, 1, im.shape) + 1j * np.random.normal(0, 1, im.shape)

# Initialize MR object with the parameters below
mr = MR_utils(tr=34.5e-3, bwpp=250e3/256, fc=127.8e6)

# Load an image and add a pilot tone to it. 
# mr.load_image(im)
mr.load_kspace(ksp)
mr.add_PT(127.8e6 + 80e3)
mr.MRshow(drng=1e-6, log=True)