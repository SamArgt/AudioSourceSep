# Statistics MSs Project: Audio Source Separation
Statistics MSc Project (2020): Audio Source Separation

github page: https://samargt.github.io/AudioSourceSep/

STATUS: WORK IN PROGRESS...

## Data

We used mixture of Piano and Violin.

The wav files can be downloaded from: ...

To Transform the wav files into melspectrograms, use  wav_to_spec.py from the pipeline module.
You will need to save the melspectrograms as tfrecords files and organize them into train/ and test/ folders in order to train the Generative Models.

You can directly download the melspectrograms in an organized folder from: ...

## Scripts

### train_ncsn.py
Script to train NCSN model 

### train_glow.py
Script to train Glow model

### train_noisy_glow.py
Script to fine-tune trained glow model at different noise levels the BASIS Separation Algorithm

### run_basis_sep.py
Script to run the BASIS Separation algorithm on the MNIST, CIFAR10 dataset or MelSpectrograms

### melspec_inversion_basis.py
Script to inverse the MelSpectrograms from BASIS back to the time domain. 

## Miscellaneous

- **train_realnvp.py**: Script to train the Real NVP model on MNIST
- **train_utils.py**: Utility functions for training
- **oracle_systems.py**: Oracle Systems for Source Separation (IBM, IRM, MWF). The Code is taken from https://github.com/sigsep/sigsep-mus-oracle
- **bss_eval_v4.py**: Evaluation of the Separation. Code Taken from https://github.com/sigsep/bsseval
- **unittest_flow_models.py**: Test the normalizing flows implementation
- **unittest_pipeline.py**: Test the pipeline module

## Modules

### pipeline module
#### preprocessing.py
Set of functions to:
- load wav files and spectrograms into tensorflow dataset
- Compute and Save spectrograms from raw audio
- Save dataset as TFRecords and Load TFRecords as dataset

#### dataloader.py

#### wav_to_spec.py

### flow_models module
Implement Normalizing flow models. Bijectors are implemented by subclassing tfp.bijector.Bijector

- **flow_builder** : build flow using Transformed Distribution from Tensorflow-probability
- **flow_glow.py** : implementation of the Glow model
- **flow_realnvp.py** implementaion of the Real NVP model
- **flow_tfp_bijectors.py** contains basic bijectors used in complex models
- **flow_tfk_layers.py** contains tf.keras.layers.Layer used for the affine coupling layers. Contains also bijectors implemented with keras (used to compare performances with the tfp implementation)
- **utils.py** : functions such as print_summary to print the trainable variables of the flow models implemented above.
- flow_tfk_models.py (deprecated) contains a keras Model class used to build a bijector from the bijectors implemented in flow_tfk_layers.py

### ncsn module
implementation of the Score Network and the Langevin Dynamics to generate samples

## References
This work is inspired by 3 main articles: the Glow model, the NCSN model and the BASIS algorithm


### The Glow paper
```bib
@inproceedings{kingma2018glow,
  title={Glow: Generative flow with invertible 1x1 convolutions},
  author={Kingma, Durk P and Dhariwal, Prafulla},
  booktitle={Advances in neural information processing systems},
  pages={10215--10224},
  year={2018}
}
```

### The NCSN paper
```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```

### The BASIS paper
```bib
@article{jayaram2020source,
  title={Source Separation with Deep Generative Priors},
  author={Jayaram, Vivek and Thickstun, John},
  journal={arXiv preprint arXiv:2002.07942},
  year={2020}
}
```



