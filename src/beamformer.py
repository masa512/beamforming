from environment import LinearArray
import torchaudio.transforms as T
from util import stft_wrapper, istft_wrapper,covariance_wrapper
import torch.linalg as linalg
import torch

#FFT PARAMS
n_fft = 512
win_length = 512
hop_length = 256

class DasBeamformer():

    def __init__(self,lin_arr:LinearArray):

        self.lin_arr = lin_arr
        self.w = None
        self.stft = T.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length,power=None)
        self.istft = T.InverseSpectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length)
    
    def eval_weight(self,theta):

        # Bring steering vectors (M,F)
        a = self.lin_arr.eval_svec(theta)
        M = a.shape[0]

        # Take conjugate transpose
        w = a.conj()
        self.w = w/M
        return self.w

    def apply_beamformer(self,theta):

        if self.w is None:
            self.w = self.eval_weight(theta) # M,F

        X = self.lin_arr.read_sensor(in_freq=True) # Returns M,F,N
        Y = torch.einsum('MF,MFN -> FN',self.w,X)

        return self.istft(Y) # Returns L
    
    def beam_pattern(self,target_theta,Ntheta):

        if self.w is None:
            self.w = self.eval_weight(target_theta) # M,F

        thetas = torch.linspace(0,180,Ntheta).tolist()

        # Get steering vectors
        s_vec_multi = self.lin_arr.eval_svec_multiangle(thetas) # T,M,F

        # Evaluate Dot product over M and leave as (T,F)
        bp = torch.log(abs(torch.einsum('MF,TMF->TF',self.w,s_vec_multi)))

        return bp.transpose(-1,-2)
        


class MvdrBeamformer():

    def __init__(self,lin_arr:LinearArray):

        self.lin_arr = lin_arr
        self.w = None
        self.stft = T.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length,power=None)
        self.istft = T.InverseSpectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length)
    
    def eval_weight(self,theta):

        # Bring steering vector (M,F)
        a = self.lin_arr.eval_svec(theta).transpose(-1,-2) # Steering vector of shape F,M where F is freq bin and M if sensor index
        
        # Spectrogram observation of interference signal (excluding source of interest)
        #X = self.lin_arr.read_sensor(theta_exclude=theta,in_freq=True) # of dimension M,F,N where N is time index
        X = self.lin_arr.read_sensor(in_freq=True)

        # Evaluate correlation (Compress in N, Expand in M)
        N = X.shape[-1]
        # Rearrange X of MFN to FMN and do X @ X.T
        X = X.transpose(0,1)
        R = 1/N * X @ X.conj().transpose(-1,-2) # Correlation of the interference
        Rinv = R.pinverse() # F,M,M

        # Apply diagonal loading to stabilize inversion
        eps = 1e-3  # Small regularization term
        R += eps * torch.eye(R.shape[-1], device=R.device)

        # Evaluate filter (Num: Compress in one of the M)
        # (Den: )
        num = torch.einsum('FM,FMM->FM', a, Rinv)
        #den = torch.einsum('FM,FMM,MF->F',a,Rinv,a.conj().transpose(-1,-2)).real.view(-1,1)
        den = torch.einsum('FM,FMM,MF->F', a, Rinv, a.conj().transpose(-1,-2)).real.view(-1,1)
        self.w = (num/den).squeeze()
        return self.w

    def apply_beamformer(self,theta):

        if self.w is None:
            self.w = self.eval_weight(theta) # M,F

        X = self.lin_arr.read_sensor(in_freq=True) # Returns M,F,N
        Y = torch.einsum('MF,MFN -> FN',self.w,X)

        return self.istft(Y) # Returns L
    
    def beam_pattern(self,target_theta,Ntheta):

        if self.w is None:
            self.w = self.eval_weight(target_theta) # M,F

        thetas = torch.linspace(0,180,Ntheta).tolist()

        # Get steering vectors
        s_vec_multi = self.lin_arr.eval_svec_multiangle(thetas) # T,M,F

        # Evaluate Dot product over M and leave as (T,F)
        bp = torch.log(abs(torch.einsum('MF,TMF->TF',self.w.transpose(-1,-2),s_vec_multi)))

        return bp.transpose(-1,-2)