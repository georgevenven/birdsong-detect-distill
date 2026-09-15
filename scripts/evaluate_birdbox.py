#!/usr/bin/env python3
"""Evaluate released BirdBox weights with native inputs and the shared benchmark."""
from birdsong_detect_distill.birdbox import MODELS, model_path, spectrogram_image, windows
from evaluate_baselines import main

if __name__ == "__main__":
    main(default_model="birdbox")
