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

        # Initialize temporal freq aspect
        self.fs = fs
        self.Nf = n_fft//2 + 1
        self.N = (L-win_length)//(hop_length) + 1

        self.stft = T.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length,power=None)
        self.istft = T.InverseSpectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length)

        self.taxis = torch.linspace(0, L/fs , steps=self.N)
        self.faxis = torch.linspace(0, fs/2, steps=self.Nf)

        # Initialize signal collection so far
        self.thetas = []
        self.signals = []

        # Noise stat
        self.snr = snr

    def __len__(self):
        return len(self.signals)

    def add_signal(self,s,theta):
        # Expects s of dimension (n_channel,N)
        assert s.size(-1) >= self.N

        # Forward spectrogram
        spectrogram = self.stft(s.squeeze())
        
        # Append to the list
        self.thetas.append(theta)
        self.signals.append(spectrogram)

    def eval_svec(self,theta):
        tau = -(self.pos.view(-1,1) * torch.sin(torch.deg2rad(torch.tensor([theta]))))/self.c # (M,1)

        # tau (M,1) and faxis (Nf)
        # (M,1) and (1,Nf)

        phi = 2*np.pi*tau * self.faxis.view(1,-1)
        s_vec = torch.exp(1j*phi) # (M,Nf)

        return s_vec # (M,Nf)

    def eval_svec_multiangle(self,thetas):
        # I want to expand this to multiple thetas (N_thetas)
        # Expects theta of int

        tau = -(self.pos.view(-1,1) @ torch.sin(torch.deg2rad(thetas.view(1,-1))))/self.c # (M,Ntheta)

        # tau (M,Ntheta) and faxis (Nf)
        # (M,Ntheta,1) and (1,1,Nf)

        phi = 2*np.pi*tau.unsqueeze(-1) * self.faxis.view(1,1,-1)
        s_vec = torch.exp(1j*phi) # (M,Ntheta,Nf)

        return s_vec # (M,Ntheta,Nf)
    
    def eval_directivity(self,thetas,theta_target):

        # Evaluatate steering vector (opposite) from all thetas (M,Nthetas,Nf)
        s_vecs_multi_angle = self.eval_svec_multiangle(thetas)
        print('Multiangle:',s_vecs_multi_angle.shape)

        # Evaluate the steering vector from the target angle (M,Nf)
        s_vec = self.eval_svec(theta_target) 
        print('Singleangle:',s_vec.shape)

        # Conjugate the first collection of the vectors and apply dot product and eval power returns (M,Nthetas)
        activation = s_vecs_multi_angle.conj().permute(dims=(1,2,0)).unsqueeze(-1).transpose(-1,-2) @ s_vec.view(s_vec.shape[0],1,s_vec.shape[1]).permute(dims=(1,2,0)).unsqueeze(-1)

        return abs(activation.squeeze())**2 # Returns power in (Nthetas,Nf)

    def read_sensor(self):
        
        # Reading on the sensor
        outputs = []

        for spectrogram,theta in zip(self.signals,self.thetas):
            # evaluate svec (M,Ntheta,Nf)
            s_vec = self.eval_svec(theta)
            # Expand dim using product s_vec : (M,Nf) and spectrogram : (Nf,N)
            # Need (M,Nf,None) and (None,Nf,N)
            stft_single_sensor_output = s_vec.view(*s_vec.shape,1) * spectrogram.view(1,*spectrogram.shape)
            # Bring back to time domain
            single_sensor_output = self.istft(stft_single_sensor_output)
            # Apply the measurement noise based on the SNR
            #Ps = abs(single_sensor_output).sum().item()**2/(self.M * self.N)
            #noise_sig = eval_noise_std(Ps)
            # Generate sensor noise and add
            outputs.append(single_sensor_output )#+ noise_sig*torch.randn_like(single_sensor_output))
        output = sum(outputs)
        return output
