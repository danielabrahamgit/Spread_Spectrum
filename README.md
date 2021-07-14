# Spread_Spectrum
### About
The goal of this repository is to provide tools to both simulate and (further down the line) generate spread spectrum modulated signals for the MRI Pilot Tone method. That is, instead of using a Pilot Tone to detect motion, we will use a spread spectrum modulation technique.    

### Simulator Usage
First off, we load MR images in the *images* directory. All images were chosen from the links listed below in the resources section. After an image is loaded, we use the MR_utils class to get it's k-space representation as well as adding a pilot tone to the k-space acquisition.

### Pilot Tone Fundementals
I also wrote a small PDF on what a pilot tone is and why it's artifact appears in the final image:
https://www.overleaf.com/read/prfxfmjvbfdy

### Resoures
**MR Image Links**
https://www.nlm.nih.gov/research/visible/image/mri/m_vm1125.t1.png
