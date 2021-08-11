# Spread_Spectrum
### About
The goal of this repository is to provide tools to both simulate and (further down the line) generate spread spectrum modulated signals for the MRI Pilot Tone method. That is, instead of using a Pilot Tone to detect motion, we will use a spread spectrum modulation technique.    

### Simulator Usage
First off, we load MR images in the *images* directory. All images were chosen from the links listed below in the resources section. After an image is loaded, we use the MR_utils class to get it's k-space representation as well as adding a pilot tone to the k-space acquisition.    

Here is the argument list for running the simulator:
```
usage: main.py [-h] [-fpt fpt] [--ssm] [--tr_rnd] [-tr tr] [-bw bw] [-fc fc]

Standard and Spread Spectrum Pilot Tone Simulator. See https://github.com/danielabrahamgit/Spread_Spectrum for documentation.

optional arguments:
  -h, --help  show this help message and exit
  -fpt fpt    The standard pilot tone frequency (assumes no SSM) (kHz). Default = 120kHz
  --ssm       Do you want to enable spread spectrum? Default = False
  --tr_rnd    Do you want to enable uncertainty in TR? Default = False
  -tr tr      Time repetition (TR): Time between readouts (ms). Default = 34ms
  -bw bw      Bandwidth: Range of frequencies in imaging band (kHz). Default = 250kHz
  -fc fc      Center frequency of scanner (MHz). Default = 127.8MHz
  ```

<u>Standard Pilot Tone Example</u>    
Suppose I would like to place a standard pilot tone at a frequency of 120kHz. Additionaly, to fully simulate the pilot tone artifact, I also want to add some uncertanty or randomness is TR. So, I would execute the following command:
```
python main.py -fpt 120 --tr_rnd
```
<u>Spread Spectrum Pilot Tone Example</u>    
When using spread spectrum, we use a default PT frequency of 0kHz. Here is an example:
```
python main.py --ssm --tr_rnd
```



### Pilot Tone Fundementals
I also wrote a small PDF on what a pilot tone is and why it's artifact appears in the final image:
https://www.overleaf.com/read/prfxfmjvbfdy

### Resoures
**MR Image Links**
https://www.nlm.nih.gov/research/visible/image/mri/m_vm1125.t1.png
