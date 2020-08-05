# Statistics MSs Project: Audio Source Separation
Statistics MSc Project (2020): Audio Source Separation

STATUS: WORK IN PROGRESS...
The README file might not be up to date. The references are missing...

## train_ncsn.py
Train NCSN model using high level Tensorflow API

## train_ncsn_custom_loop.py
Train NCSN model using custom loop

### train_flow.py

```bash
python train_flow.py
Train Flow model on MNIST or CIFAR10 dataset or MelSpectrograms Data

The script uses Tensorboard to visualize the loss and samples during training. To launch tensorboard:
```bash
cd OUTPUT
tensorboard --logdir tensorboard_logs
```
## train_noisy_glow.py

```bash
python noise_conditioned_models.py
```
Train glow noise conditioned models for the BASIS Separation Algorithm

## run_basis_sep.py
Run the BASIS Separation algorithm on the MNIST or CIFAR10 dataset

## pipeline module
Set of functions to:
- load wav files and spectrograms into tensorflow dataset
- Compute and Save spectrograms from raw audio
- Save dataset as TFRecords and Load TFRecords as dataset

## flow_models module
Implement Normalizing flow models. Bijectors are implemented by subclassing tfp.bijector.Bijector

- **flow_builder** : build flow using Transformed Distribution from Tensorflow-probability
- **flow_glow.py** : implementation of the Glow model
- **flow_realnvp.py** implementaion of the Real NVP model
- **flow_tfp_bijectors.py** contains basic bijectors used in complex models
- **flow_tfk_layers.py** contains tf.keras.layers.Layer used for the affine coupling layers. Contains also bijectors implemented with keras (used to compare performances with the tfp implementation)
- **utils.py** : functions such as print_summary to print the trainable variables of the flow models implemented above.
- flow_tfk_models.py (deprecated) contains a keras Model class used to build a bijector from the bijectors implemented in flow_tfk_layers.py



