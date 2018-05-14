# encoding: utf-8
import tensorflow as tf
from hparams import hparams

"""
TensorFlow implementation of Griffin-lim Algorithm for voice reconstruction
"""


# TF
def spectrogram2wav(spectrogram, n_iter=hparams.griffin_lim_iters, n_fft=(hparams.num_freq - 1) * 2,
                    win_length=int(hparams.frame_length_ms / 1000 * hparams.sample_rate),
                    hop_length=int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)):
    '''Converts spectrogram into a waveform using Griffin-lim's raw.
    '''

    def invert_spectrogram(spectrogram):
        '''
        spectrogram: [t, f]
        '''
        spectrogram = tf.expand_dims(spectrogram, 0)
        inversed = tf.contrib.signal.inverse_stft(spectrogram, win_length, hop_length, n_fft)
        squeezed = tf.squeeze(inversed, 0)
        return squeezed

    spectrogram = tf.transpose(spectrogram)

    spectrogram = tf.cast(spectrogram, dtype=tf.complex64)  # [t, f]
    X_best = tf.identity(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best)
        est = tf.contrib.signal.stft(X_t, win_length, hop_length, n_fft, pad_end=False)  # (1, T, n_fft/2+1)
        phase = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)  # [t, f]
        X_best = spectrogram * phase  # [t, t]
    X_t = invert_spectrogram(X_best)
    y = tf.real(X_t)

    return y


def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    return _inv_preemphasis(spectrogram2wav(S ** hparams.power))  # Reconstruct phase


def _denormalize(D):
    return (((tf.clip_by_value(D, -hparams.max_abs_value,
                               hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (
                     2 * hparams.max_abs_value)) + hparams.min_level_db)


def _db_to_amp(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _inv_preemphasis(x):
    N = tf.shape(x)[0]
    i = tf.constant(0)
    W = tf.zeros(shape=tf.shape(x), dtype=tf.float32)

    def condition(i, y):
        return tf.less(i, N)

    def body(i, y):
        tmp = tf.slice(x, [0], [i + 1])
        tmp = tf.concat([tf.zeros([N - i - 1]), tmp], -1)
        y = hparams.preemphasis * y + tmp
        i = tf.add(i, 1)
        return [i, y]

    final = tf.while_loop(condition, body, [i, W])

    y = final[1]

    return y
