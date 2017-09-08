# encoding: utf-8
from utils import audio
from hparams import hparams
import numpy as np
import io
from griffin_lim import inv_spectrogram, tf

if __name__ == '__main__':
    wavs = ["data/000001.wav", "data/000002.wav"]
    outputs_py = ["data/000001.gen.wav", "data/000002.gen.wav"]
    outputs_tf = ["data/000001.gen.tf.wav", "data/000002.gen.tf.wav"]
    wavs = [audio.load_wav(wav_path, hparams.sample_rate) for wav_path in wavs]
    spectrogram = [audio.spectrogram(wav).astype(np.float32) for wav in wavs]
    print(spectrogram[0].shape)
    print(spectrogram[1].shape)

    # --------------------------------- librosa Version ---------------------------------
    # convert back
    gens = [audio.inv_spectrogram(s) for s in spectrogram]

    for gen, output in zip(gens, outputs_py):
        out = io.BytesIO()
        audio.save_wav(gen, out)

        with open(output, "wb") as f:
            f.write(out.getvalue())

    # --------------------------------- TensorFlow Version ---------------------------------

    samples = [inv_spectrogram(spec) for spec in spectrogram]

    with tf.Session() as sess:
        samples = [sess.run(sample) for sample in samples]

    for gen, output in zip(samples, outputs_tf):
        out = io.BytesIO()
        audio.save_wav(gen, out)

        with open(output, "wb") as f:
            f.write(out.getvalue())
