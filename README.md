# Statistics MSs Project: Audio Source Separation
Statistics MSc Project (2020): Audio Source Separation

### mnist_train_tfp.py

Script to train a Normalizing Flow model on the MNIST dataset

```bash
python mnist_train_tfp.py OUPUT N_EPOCHS
```
OUTPUT: directory where to save the log, the loss history, the variables of the model and some samples
N_EPOCHS: number of epochs to train the model

### mnist_train_tfk.py
(deprecated)
Script to train a Normalizing Flow model using the keras implementation of the bijectors


## pipeline
Functions and Scripts to load and preprocess the data:
- Generating equal length sequences from raw audio
- mel spectrograms transformation
- loading into tensorflow dataset type

## flow_models
Normalizing Flow models implemented as tensorflow keras model
- Real NVP


