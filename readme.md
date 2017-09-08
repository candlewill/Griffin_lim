# Griffin Lim

This repo is a TensorFlow implementation of Griffin-Lim algorithm for voice reconstruction.

For comparision, a librosa implementation version is also included in this repo.

### Dependencies

* TensorFlow  >=1.3
* Python 3.x
* scipy
* numpy
* librosa

### Run
To test the performance, we could execute the following command:
```
python test.py
```

The `test.py` script does the following things in order:
1. Extract the spectrogram features from waves
2. Reconstruction voice using Griffin-lim algorithm

### Data
All data used is in the `data` folder. If we want to use other data, replace it.

### Acknowledgement
Some code are borrowed from the following repo:

* [Kyubyong/tensorflow-exercises](https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb)
* [keithito/tacotron](https://github.com/keithito/tacotron)

### Contact
[Yunchao He](yunchaohe@gmail.com)