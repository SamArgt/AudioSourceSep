# MSc Statistics Project - Audio Source Separation

## NCSN Results

### Piano

![Spectrograms](https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/ckpt10_generated_samples.png)

<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_1.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_2.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_3.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_4.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_5.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_6.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_7.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_piano_192_32_dB_custom_loop/generated_samples/inv_gen_sample_8.wav" type="audio/wav">
</audio>

### Violin

![Spectrograms](https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/ckpt10_generated_samples.png)

<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_1.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_2.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_3.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_4.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_5.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_6.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_7.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/trained_ncsn/ncsn_violin_192_32_dB_custom_loop/generated_samples/inv_gen_sample_8.wav" type="audio/wav">
</audio>


## BASIS Separation

Beethoven Sonata 1: 1st minute. The generative model used is the NCSN.
The original sources and mixture are obtained with the same method as the estimated sources, i.e., inversion of the melspectrograms.

### Originals Sources

<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/basis_sep_results/beethoven_sonata_1_sep_1min/gt1.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/basis_sep_results/beethoven_sonata_1_sep_1min/gt2.wav" type="audio/wav">
</audio>

### Mixture

<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/basis_sep_results/beethoven_sonata_1_sep_1min/mix.wav" type="audio/wav">
</audio>

### Estimated Sources

<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/basis_sep_results/beethoven_sonata_1_sep_1min/reuse_phase/sep1.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/basis_sep_results/beethoven_sonata_1_sep_1min/reuse_phase/sep2.wav" type="audio/wav">
</audio>

After applying a Single-channel Wiener Filter
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/basis_sep_results/beethoven_sonata_1_sep_1min/wiener/sep1.wav" type="audio/wav">
</audio>
<audio controls preload="auto">
<source src="https://raw.githubusercontent.com/SamArgt/AudioSourceSep/master/basis_sep_results/beethoven_sonata_1_sep_1min/wiener/sep2.wav" type="audio/wav">
</audio>
