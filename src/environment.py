import math
import numpy as np
import torch
import torchaudio.transforms as T
from util import eval_noise_std
############ For Environment ###############
speed_of_light = 343

class LinearArray():

    def __init__(self,M,L,d,fs,n_fft = 512,win_length = 512,hop_length = 256,snr = math.inf):
        # Initializes the geometry
        self.c = speed_of_light
        self.M = M # n of sensors
        self.d = d # Spacing bw each
        self.pos = torch.Tensor([i*d for i in range(M)])
        self.pos = self.pos - self.pos[len(self.pos)//2] # (M,)

        # Initialize temporal freq aspect
        self.fs = fs
        self.Nf = n_fft//2 + 1
        self.N = (L-win_length)//(hop_length) + 1

        # STFT,ISTFT Params
        self.n_fft = n_fft
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length)
        self.hop_length = hop_length

        self.taxis = torch.linspace(0, L/fs , steps=self.N)
        self.faxis = torch.linspace(0, fs/2, steps=self.Nf)

        # Initialize signal collection so far
        self.thetas = []
        self.spectrograms = []

        # Noise stat
        self.snr = snr

    def __len__(self):
        return len(self.spectrograms)

    def add_signal(self,s,theta):
        # Expects s of dimension (N)
        # Forward spectrogram
        spectrogram = torch.stft(s,self.n_fft,self.hop_length,self.win_length,window = self.window,return_complex=True)
        
        # Append to the list
        self.thetas.append(theta)
        self.spectrograms.append(spectrogram)

        print(s.shape,spectrogram.shape)

    def eval_svec(self,theta):
        theta_rad = torch.deg2rad(torch.tensor([theta]))
        tau = (self.pos * torch.sin(theta_rad))/self.c # (M,)
        # tau (M) and faxis (Nf)

        faxis = torch.linspace(0,self.fs//2,self.Nf)
        phi = 2*np.pi * torch.einsum('M,F -> MF',tau,faxis)             
        s_vec = torch.exp(1j*phi) # (M,Nf)

        return s_vec # (M,Nf)

    def eval_svec_multiangle(self,thetas):
        # I want to expand this to multiple thetas (list of length thetas)
        # Radian Angles
        thetas_rad = torch.deg2rad(torch.tensor(thetas)) # Size (T)
        # Freq Axis
        faxis = torch.linspace(0,self.fs//2,self.Nf) # Size (F)
        tau = torch.einsum('T,M -> TM',torch.sin(thetas_rad),self.pos)/self.c
        # self.pos has size (M) - We want T,M,F
        phi = 2*np.pi * torch.einsum('TM,F -> TMF',tau,faxis)
        s_vec = torch.exp(1j*phi)

        return s_vec # (Ntheta,M,Nf)
    
    def read_sensor(self,theta_exclude = None, in_freq = False):

        thetas = self.thetas[:]
        spectrograms = self.spectrograms[:]

        if theta_exclude is not None:
            # Remove unwanted theta/signal pair
            idx = thetas.index(theta_exclude)
            _ = thetas.pop(idx)
            _ = spectrograms.pop(idx)

        print(thetas)

        # Steering vector
        s_vec_multi = self.eval_svec_multiangle(thetas) # (T,M,F)

        # Build a torch tensor for signals (T,F,N) 
        S  = torch.stack(spectrograms,dim=0)
        print(S.shape,s_vec_multi.shape)

        
        # Mul and summation along T and element wise on the F and keeep the rest 
        sensor_output = torch.einsum('TMF,TFN -> MFN', s_vec_multi,S)
        
        if not in_freq:

            sensor_output = torch.istft(sensor_output,self.n_fft,self.hop_length,self.win_length,window=self.window,return_complex=False)

        return sensor_output
