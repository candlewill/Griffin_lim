import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Audio:
    num_mels=80,
    num_freq=513,
    sample_rate=22050,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    max_abs_value=4,
    power=1.5,
    fft_size=1024,
    hop_size=256,

    # Eval:
    griffin_lim_iters=60
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
