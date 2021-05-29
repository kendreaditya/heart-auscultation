<div align="center">

# Employing Adversarial Machine Learning and Computer Audition for Smartphone-Based

Real-Time Arrhythmia Classification in Heart Sounds

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)

<!--
Conference
-->

</div>

## Description

We propose a novel approach to detect arrhythmias in Phonocardiograms (PCGs). Typically, many arrhythmia conditions are unknown until a patient is suggested an ECG/EKG test. This method, despite being accurate, limits the use case to hospitals and clinics with specialized equipment; thus, limiting the portability of diagnosing. Implementation of Adversarial Machine Learning (ML) and Computer Audition (CA) in combination with heart sounds provide ease of access to everyone who has a device capable of recording audio. Ideally, allowing medical professionals to treat arrhythmias in the developmental stages. The new design is comprised of two subsystems: one is based on the relationship between Electrocardiograms (ECGs) and PCGs, and the other between PCGs and arrhythmias. The first subsystem uses a Generative Adversarial Networks (GAN), in which both generated and real PCG signals are fed into the discriminator for classification. In subsystem two, ECG spectrograms are dimensionally reduced, then constructed into PCG spectrograms using a transGAN. These constructed PCG spectrograms, when converted back into time series, should be identical to the ground truth. This novel approach allows for an increase in the number of cardiovascular pathologies classified in heart sounds.

## How to run

First, install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project
cd deep-learning-project-template
pip install -e .
pip install -r requirements.txt
```

Next, navigate to any file and run it.

```bash
# module folder
cd project

# run module (example: mnist as your main contribution)
python lit_classifier_main.py
```

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
