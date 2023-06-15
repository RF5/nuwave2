dependencies = ['torch', 'torchaudio', 'numpy', 'scipy', 'omegaconf', 'prefetch_generator']


from lightning_model import NuWave2
from omegaconf import OmegaConf as OC
from pathlib import Path
from torch import Tensor
import torchaudio
import torch

from utils.stft import STFTMag

from scipy.signal import resample_poly


class NuWave2Wrapper():

    def __init__(self, model, hparams, hi_cutoff, input_sr=16000, device='cpu') -> None:
        self.model = model
        self.hparams = hparams
        self.input_sr = input_sr
        self.hi_cutoff = hi_cutoff
        self.device = device
        self.out_sr = hparams.audio.sampling_rate

    @torch.inference_mode()
    def infer(self, input: str | Tensor | Path):
        """ Perform inference on mono audio of shape (n_channels, T), returning result in format (1, T). """
        steps = 8
        noise_schedule = eval(self.hparams.dpm.infer_schedule)
        if type(input) in [str, Path]:
            wav, inp_sr = torchaudio.load(input)
            assert inp_sr == self.input_sr, f"input audio's sample rate {inp_sr} is not the expected {self.input_sr}"
        else: wav = input
        assert wav.shape[0] == 1, f"Only mono audio input currently supported!"
        wav: Tensor = wav / wav.abs().max()
        wav = wav.squeeze().cpu().numpy()
        # wav, _ = librosa.load('original.wav', sr=self.input_sr, mono=True)
        # wav /= np.max(np.abs(wav))

        # upsample to the original sampling rate
        wav_l = resample_poly(wav, self.hparams.audio.sampling_rate, self.input_sr)
        wav_l = wav_l[:len(wav_l) - len(wav_l) % self.hparams.audio.hop_length]

        fft_size = self.hparams.audio.filter_length // 2 + 1
        band = torch.zeros(fft_size, dtype=torch.int64)
        band[:int(self.hi_cutoff * fft_size)] = 1

        wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(self.device)
        band = band.unsqueeze(0).to(self.device)
        wav_recon, wav_list = self.model.inference(wav_l, band, steps, noise_schedule)

        wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps).cpu()
        return wav_recon

def nuwave2_16khz(pretrained=True, progress=True, device='cuda') -> NuWave2Wrapper:
    """ Load pretrained nuwave2 model. """
    cp = Path(__file__).parent.absolute()

    hparams = OC.load(str(cp/'hparameter.yaml'))
    model = NuWave2(hparams).to(device)
    model.eval()
    stft = STFTMag()
    sr = 16000
    highcut = sr // 2
    nyq = 0.5 * hparams.audio.sampling_rate
    hi = highcut / nyq
    
    if pretrained:
        url = "https://github.com/RF5/nuwave2/releases/download/v1.0/nuwave2.ckpt"
        ckpt = torch.hub.load_state_dict_from_url(
            url,
            map_location=device,
            progress=progress
        )
        model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"[nu-wave2] model loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    wrapper = NuWave2Wrapper(model, hparams, hi, input_sr=sr, device=device)
    return wrapper
