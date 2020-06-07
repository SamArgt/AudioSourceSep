# Statistics MSs Project: Audio Source Separation
Statistics MSc Project (2020): Audio Source Separation

### mnist_train_tfp_distributed.py

```bash
python mnist_train_tfp_distributed.py [-h] [--output OUTPUT]
                                      [--n_epochs N_EPOCHS]
                                      [--restore RESTORE] [--K K]
                                      [--batch_size BATCH_SIZE]
                                      [--n_filters N_FILTERS]
                                      [--use_logit USE_LOGIT]
                                      [--learning_rate LEARNING_RATE]
```
Train Flow model on MNIST dataset

optional arguments: <br />
  >-h, --help            show this help message and exit<br />
  >--output OUTPUT       output dirpath for savings<br />
  >--n_epochs N_EPOCHS   number of epochs to train<br />
  >--restore RESTORE     directory of saved weights (optional)<br />
  >--K K                 Number of Step of Flow in each Block<br />
  >--batch_size BATCH_SIZE<br />
  >--n_filters N_FILTERS<br />
  >>                       number of filters in the Convolutional Network<br />
  >--use_logit USE_LOGIT<br />
  >>                      Either to use logit function to preprocess the data<br />
  >--learning_rate LEARNING_RATE<br />

The script uses Tensorboard to visualize the loss and samples during training. To launch tensorboard:
```bash
cd OUTPUT
tensorboard --logdir tensorboard_logs
```
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



