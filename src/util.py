import torchaudio.transforms as T
import torch

def eval_noise_std(Ps,snr):
        
        return Ps * 10**(-snr/20)


def stft_wrapper(n_fft = 512,win_length = 512,hop_length = 256):

   stft = T.Spectrogram(n_fft = n_fft,win_length = win_length,hop_length = hop_length, power=None)
   return stft

def istft_wrapper(n_fft = 512,win_length = 512,hop_length = 256):
      
      istft = T.InverseSpectrogram(n_fft = n_fft,win_length = win_length,hop_length = hop_length)
      return istft

def covariance_wrapper(X:torch.Tensor,Y:torch.Tensor = None):
        # Assume X and Y are each of size (*shape[:-2],N,d)
      
        # Cross Correlation if Y is not None
        if Y is None:
                
                Y = X
        
        assert X.shape[-2:] == Y.shape[-2:], "Dimensions Not matching up"


        N = X.shape[-2]
        return 1/N * X.transpose(-1,-2) @ Y



