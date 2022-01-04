# Masters degree project

# Goals
The goal that is meant to be accomplished is yet for debate but I restricted the choice to 2 options:

## Music transcription
The model would convert a music track containing only one instrument to data representing classical musical notation. The program initially would **require the user to provide the tempo** and the tracks would need to start with the start of a beat.

This problem would benefit greatly from a **recurrent neural network**, since music often follows patterns(scales, modes), which would be interesting to implement as a stacked model.

However the problem might be too simple. If we require that the composition is just singular notes instead of chords then the problem is reduced to simple pitch reading and that is too trivial for ML use. If we instead allow more freedom then the problem described in the next paragraf would be a component of this one so then it might become too complex.

## Chord identification
There are many chords that can be played on a guitar. This many:
![guitar_chords][chords]
600. Put sideways for convinience ;)

And many of these chords have a couple of variations. Add the fact that a chord can be played strumming up, down or even with finger-style we have an array of different sounds constituting a single chord. So in a way it is more of a classification problem rather than identification.

The program would get a sound file with a chord being played and return its name. Simple as that.

# Data
Here i list datasets that might be usefull in the project. The datasets are from reputable sources and seem to be well constructed. If all else fails there is still the option of making my own data.
## [IDMT-SMT-Guitar](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html)

## [IDMT-SMT-Chords](https://www.idmt.fraunhofer.de/en/publications/datasets/chords.html)

## [Guitarset: A dataset for guitar transcription](https://zenodo.org/record/3371780#.YdSZmooo-V4)

## Selfmade


# Technologies to be used
## Python
Great ML frameworks. A lot of audio, data manipulation and scientific libraries. Posibilitty to work in notebook mode. Preety selfexplainatory.

## TensorFlow
I chose TensorFlow as the framework for my model building needs. I was split between TensorFlow and PyTorch but i decided on TF due to a higher amount of material using TF as well as the fact that i have greater experience with it rather than PyTorch.

## Librosa
Librosa is a library for audio analysis in python. The library provides such functionalities as **chromagram extraction** and other audio representations.

# Bibliography
1. Intro to stacked networks ([machinelearningmastery.com/...](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/))
2. [Chord recognition using Fourier transformation](https://onlinelibrary.wiley.com/doi/epdf/10.1002/tee.23492)

# Notes
1. Make some proofs of concept

[chords]: readme_resources/guitar_chords.png "Guitar chords n=600"