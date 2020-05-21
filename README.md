# Statistics MSs Project: Audio Source Separation
Statistics MSc Project (2020): Audio Source Separation

### mnist_train_tfp.py

Script to train a Normalizing Flow model on the MNIST dataset

```bash
python mnist_train_tfp.py OUTPUT --n_epochs N_EPOCHS
```
OUTPUT: directory where to save the log, the loss history, the variables of the model and some samples
N_EPOCHS: number of epochs to train the model

The script uses Tensorboard to visualize the loss and samples during training. To launch tensorboard:
```bash
cd OUTPUT
tensorboard --logdir tensorboard_logs/gradient_tape 
```

### mnist_train_tfk.py
(deprecated)
Script to train a Normalizing Flow model using the keras implementation of the bijectors


## pipeline module
Set of functions to:
- load wav files and spectrograms into tensorflow dataset
- Compute and Save spectrograms from raw audio
- Save dataset as TFRecords and Load TFRecords as dataset

## flow_models module
Implement Normalizing flow models. Bijectors are implemented by subclassing tfp.bijector.Bijector

- **flow_glow.py** : implementation of the Glow model
- **flow_realnvp.py** implementaion of the Real NVP model
- **flow_tfp_bijectors.py** contains basic bijectors used in complex models
- **flow_tfk_layers.py** contains tf.keras.layers.Layer used for the affine coupling layers. Contains also bijectors implemented with keras (used to compare performances with the tfp implementation)
- **test_flow_tfp_bijectors.py** unittest for the custom bijectors implemented in the files above.
- **utils.py** : functions such as print_summary to print the trainable variables of the flow models implemented above.
- flow_tfk_models.py (deprecated) contains a keras Model class used to build a bijector from the bijectors implemented in flow_tfk_layers.py



