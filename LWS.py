# encoding: utf-8
from utils import audio
from hparams import hparams
import numpy as np
import os
import lws


def main():
    data_foler = "data"
    wavs = [os.path.join(data_foler, file[:-4]) for file in os.listdir(data_foler) if file.endswith(".wav")]
    outputs_lws = [file + ".lws.gen.wav" for file in wavs]
    wavs = [audio.load_wav(wav_path + ".wav", hparams.sample_rate) for wav_path in wavs]

    lws_processor = lws.lws(512, 128, mode="speech")  # 512: window length; 128: window shift
    i = 0
    for x in wavs:
        X = lws_processor.stft(x)  # where x is a single-channel waveform
        X0 = np.abs(X)  # Magnitude spectrogram
        print('{:6}: {:5.2f} dB'.format('Abs(X)', lws_processor.get_consistency(X0)))
        X1 = lws_processor.run_lws(
            X0)  # reconstruction from magnitude (in general, one can reconstruct from an initial complex spectrogram)
        print(X1.shape)
        print('{:6}: {:5.2f} dB'.format('LWS', lws_processor.get_consistency(X1)))
        print(X1.shape)
        wav = lws_processor.istft(X1).astype(np.float32)

        audio.save_wav(wav, outputs_lws[i])
        i += 1


if __name__ == '__main__':
    main()
