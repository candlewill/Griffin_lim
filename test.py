# encoding: utf-8
from utils import audio
from hparams import hparams
import numpy as np
from griffin_lim import inv_spectrogram, tf
import os

if __name__ == '__main__':
    data_foler = "data"
    wavs = [os.path.join(data_foler, file[:-4]) for file in os.listdir(data_foler) if file.endswith(".wav")]
    outputs_py = [file + ".py.gen.wav" for file in wavs]
    outputs_tf = [file + ".tf.gen.wav" for file in wavs]
    wavs = [audio.load_wav(wav_path + ".wav", hparams.sample_rate) for wav_path in wavs]
    spectrogram = [audio.spectrogram(wav).astype(np.float32) for wav in wavs]
    print("Linear spectrograms dim: ")
    print(spectrogram[0].shape)
    # --------------------------------- librosa Version ---------------------------------
    # convert back
    gens = [audio.inv_spectrogram(s) for s in spectrogram]

    for gen, output in zip(gens, outputs_py):
        audio.save_wav(gen, output)

    # --------------------------------- TensorFlow Version ---------------------------------

    samples = [inv_spectrogram(spec) for spec in spectrogram]

    with tf.Session() as sess:
        samples = [sess.run(sample) for sample in samples]

    for gen, output in zip(samples, outputs_tf):
        audio.save_wav(gen, output)

    print("Done!")
