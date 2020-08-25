# Statistics MSs Project: Audio Source Separation
Statistics MSc Project (2020): Audio Source Separation

github page: https://samargt.github.io/AudioSourceSep/

STATUS: WORK IN PROGRESS...
The README file might not be up to date. The references are missing...

## train_ncsn.py
Script to train NCSN model 

## train_glow.py
Script to train Glow model

## train_noisy_glow.py
Script to fine-tune trained glow model at different noise levels the BASIS Separation Algorithm

## run_basis_sep.py
Script to run the BASIS Separation algorithm on the MNIST, CIFAR10 dataset or MelSpectrograms

## pipeline module
### preprocessing.py
Set of functions to:
- load wav files and spectrograms into tensorflow dataset
- Compute and Save spectrograms from raw audio
- Save dataset as TFRecords and Load TFRecords as dataset

### dataloader.py

### wav_to_spec.py

## flow_models module
Implement Normalizing flow models. Bijectors are implemented by subclassing tfp.bijector.Bijector

- **flow_builder** : build flow using Transformed Distribution from Tensorflow-probability
- **flow_glow.py** : implementation of the Glow model
- **flow_realnvp.py** implementaion of the Real NVP model
- **flow_tfp_bijectors.py** contains basic bijectors used in complex models
- **flow_tfk_layers.py** contains tf.keras.layers.Layer used for the affine coupling layers. Contains also bijectors implemented with keras (used to compare performances with the tfp implementation)
- **utils.py** : functions such as print_summary to print the trainable variables of the flow models implemented above.
- flow_tfk_models.py (deprecated) contains a keras Model class used to build a bijector from the bijectors implemented in flow_tfk_layers.py

## ncsn module
Implement the Score Network and the Langevin Dynamics to generate samples



