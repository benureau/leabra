#!/bin/bash

mkdir -p outs
jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True ../notebooks/neuron.ipynb     --output ../tests/outs/neuron.html
jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True ../notebooks/noisy_xx1.ipynb  --output ../tests/outs/noisy_xx1.html
