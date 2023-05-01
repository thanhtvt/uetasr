<div align="center">

<img src="./docs/images/logo.png" width=170>

<h1> UETASR

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tensorflow](https://img.shields.io/badge/TensorFlow_2.8_%7C_2.9_%7C_2.10-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/thanhtvt/uetasr)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

</h1>

<h2> An Automatic Speech Recognition toolkit in TensorFlow 2 </h2>

_Suggestions are always welcome!_

</div>

<br>

## Key features

UETASR provides various useful tools to speed up and facilitate research on speech technologies:

- A [YAML-based hyperparameter](https://github.com/speechbrain/HyperPyYAML) specification language that describes all types of hyperparameters, from individual numbers to complete objects.

- Single and Multi-GPUs training and inference with TensorFlow 2 Data-Parallel or Distributed Data-Parallel.

- A transparent and entirely customizable data input and output pipeline, enabling users to customize the I/O pipelines.

- Logging and visualization with [WandB](https://wandb.ai) and [TensorBoard](https://www.tensorflow.org/tensorboard).

- Error analysis tools to help users debug their models.

### Supported Models

- CTC and Transducer architectures with any encoders (and decoders) can be plugged into the framework.
- Gradient Accumulation for large batch training is supported.
- Currently supported:
    + Conformer (https://arxiv.org/abs/2005.08100)
    + Emformer (https://arxiv.org/abs/2010.10759)

### Feature extraction and augmentation

UETASR provides efficient and GPU-friendly **on-the-fly** speech augmentation pipelines and acoustic feature extraction:
- Augmentation:
    + Adaptive SpecAugment ([paper](https://arxiv.org/abs/1912.05533))
    + SpliceOut ([paper](https://arxiv.org/abs/2110.00046))
    + Gain, Time Stretch, Pitch Shift, etc. ([paper](https://www.isca-speech.org/archive/interspeech_2015/ko15_interspeech.html))
- Featurization:
    + MFCC, Fbank, Spectrogram, etc.
    + Subword tokenization (BPE, Unigram, etc.)

## Installation

For training and testing, you can use `git clone` to install some optional packages from other authors (`ctc_loss`, `rnnt_loss`, etc.)

### Prerequisites

- TensorFlow >= 2.9.0

- CuDNN >= 8.1.0

- CUDA >= 11.2

- Nvidia driver >= 470

### Install with GitHub

Once you have created your Python environment (Python 3.6+) you can simply type:

```bash
git clone https://github.com/thanhtvt/uetasr.git
cd uetasr
pip install -e .
```

Then you can access uetasr with:

```python
import uetasr
```

### Install with Conda

```bash
git clone https://github.com/thanhtvt/uetasr.git

conda create --name uetasr python=3.8
conda activate uetasr
conda install cudnn=8.1.0

cd uetasr

pip install -e .
```

### Install with Docker

Build docker from Dockerfile:

```bash
docker build -t uetasr:v1.0.0 .
```

Run container from `uetasr` image:

```bash
docker run -it --name uetasr --gpus all -v <workspace_dir>:/workspace uetasr:v1.0.0 bash
```

## Getting Started
1. Define config YAML file, see the `config.yaml` file [this folder](./egs/vlsp2022/conformer/v3/) for reference.
2. Download your corpus and create a script to generate the .tsv file (see [this file](./egs/vlsp2022/conformer/data/vlsp2022-labeled.tsv) for reference). Check our provided [`tools`](./tools) whether they meet your need.
3. Create `transcript.txt` and `cmvn.tsv` files for your corpus. We implement [this script](./tools/cmvn_transcript_generator.py) to generate those files, knowing the .tsv file generated in step 2.
4. For training, check `train.py` in the `egs` folder to see the options.
5. For testing, check `test.py` in the `egs` folder to see the options.
6. For evaluating and error analysis, check `asr_evaluation.py` in the `tools` folder to see the options.
7. [Optional] To publish your model on 🤗, check this [space](https://huggingface.co/spaces/thanhtvt/uetasr) for reference.

## References & Credits
1. [namnv1906](https://github.com/namnv1906/) (for the guidance & initial version of this toolkit)
2. [TensorFlowASR: Almost State-of-the-art Automatic Speech Recognition in Tensorflow 2](https://github.com/TensorSpeech/TensorFlowASR)
3. [ESPNet: End-to-End Speech Processing Toolkit](https://github.com/espnet/espnet)
4. [SpeechBrain: A PyTorch-based Speech Toolkit](https://github.com/speechbrain/speechbrain)
5. [Python module for evaluting ASR hypotheses](https://github.com/belambert/asr-evaluation)
6. [Accumulated Gradients for TensorFlow 2](https://github.com/andreped/GradientAccumulator)