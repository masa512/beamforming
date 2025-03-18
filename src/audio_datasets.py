from torch.utils.data import Dataset
import os
import librosa
class YesNoDataset(Dataset):

    def __init__(self,audio_path):

        super().__init__()

        self.audio_path = audio_path
        self.filelist = [fname for fname in os.listdir(audio_path) if fname.endswith('.wav')]

    def __len__(self):

        return len(self.filelist)
    
    def __getitem__(self, index):

        signal, sample_rate = librosa.load(os.path.join(self.audio_path,self.filelist[index]), sr=None) #sr=None preserves original sample rate
        
        return (signal.squeeze(),sample_rate)
    