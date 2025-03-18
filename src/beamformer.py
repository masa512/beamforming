from environment import LinearArray
import torchaudio.transforms as T
from util import stft_wrapper, istft_wrapper,covariance_wrapper
import torch.linalg as linalg
import torch

def apply_beamformer(X, w, n_fft = 512,win_length = 512,hop_length = 256):
    # X has shape (M,L)
    # w has shape (Nf,M)

    M = X.shape[0] # M

    # SFTF and ISTFT
    stft = T.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length,power=None)
    istft = T.InverseSpectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length)

    # Apply STFT on the Sensor signal
    spectrogram = stft(X) # M,Nf,N

    # Apply the weight
    filtered_spectrogram= 1/M * (w.transpose(-1,-2).unsqueeze(-1) * spectrogram).sum(dim = 0, keepdim = False)
    
    # ISTFT
    output_signal = istft(filtered_spectrogram)
    return output_signal


def get_directivity(lin_arr:LinearArray,w:torch.Tensor,n_thetas):
    # w has shape (Nf,M) -> (1,Nf,M)
    # Thetas
    thetas = torch.linspace(0,180,n_thetas)
    # Evaluatate steering vector (opposite) from all thetas (M,Nthetas,Nf)-> (Nthetas,Nf,M)
    s_vecs_multi_angle = lin_arr.eval_svec_multiangle(thetas)
    # Apply the weight w on above
    print(s_vecs_multi_angle.shape, w.shape)
    activation = s_vecs_multi_angle.permute(1,2,0) * w.squeeze(0)

    return abs(activation.sum(dim=-1))**2 # Returns power in (Nthetas,Nf)

def DAS_beamformer(lin_arr:LinearArray,theta_target):

    # Extract the steering vec and conj
    s_vec = lin_arr.eval_svec(theta_target)

    return s_vec.conj().transpose(-1,-2)


def MVDR_beamformer(lin_arr:LinearArray,theta_target):

    # Extract the steering vec and conj
    s_vec = lin_arr.eval_svec(theta_target).conj() # (M,Nf)

    # Evaluate the inverse correlation function of the observation
    sensor_out = lin_arr.read_sensor()

    # The observation must be in dimension (M,N) Take stft
    stft = stft_wrapper()
    sensor_spectrogram = stft(sensor_out)

    # Evaluate covariance only along the last axis (Sensor spectrogram has dim (M,Nf,N)
    R = covariance_wrapper(sensor_spectrogram.permute((1,2,0))) #(Nf,M,M)

    # MVDR beamformer is defined as... (R^-1 a)/(a.H R^-1 a)
    Rinv = linalg.pinv(R) # (Nf,M,M)
    a = s_vec.transpose(-1,-2).unsqueeze(-1) # (Nf,M,1)
    ah = a.transpose(-1,-2).conj() # (Nf,1,M)
    w = (Rinv @ a)/(ah @ Rinv @ a)

    return w.squeeze() # (Nf,M)
    
    