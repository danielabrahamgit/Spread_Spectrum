import nrrd
import numpy as np
import matplotlib.pyplot as plt
from utils import sig_utils, MR_utils
from PIL import Image

# Load MR image
im = np.array(Image.open('images/brain.png'))

# Initialize MR object with the parameters below
mr = MR_utils(tr=34e-3, bwpp=250e3/256, fc=127.8e6)

# Load an image and add a pilot tone to it. 
mr.load_image(im)
mr.add_PT(127.8e6 + 120e3)
mr.MRshow()