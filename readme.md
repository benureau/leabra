# Leabra

[![Binder](http://mybinder.org/badge.svg)](https://beta.mybinder.org/v2/gh/benureau/leabra/master)  [![Build Status](https://travis-ci.org/benureau/leabra.svg?branch=master)](https://travis-ci.org/benureau/leabra)

This repository holds a Python implementation of the [Leabra](https://grey.colorado.edu/emergent/index.php/Leabra) (Local, Error-driven and Associative, Biologically Realistic Algorithm) framework. The reference implementation for Leabra is in [emergent](https://grey.colorado.edu/emergent/index.php/Main_Page) developped by the [Computational Cognitive Neuroscience Laboratory](https://grey.colorado.edu/CompCogNeuro/index.php/CCNLab) at the [University of Colorado Boulder](http://www.colorado.edu/). This Python implementation targets [emergent 8.1.0](https://grey.colorado.edu/emergent/index.php/Changes_8.1.0), and only implements the rate-coded mode —which includes some spiking behavior, but is different from the discrete spiking mode (which is not implemented).

This work is the fruit of the collaboration of the [Computational Cognitive Neuroscience Laboratory](https://grey.colorado.edu/CompCogNeuro/index.php/CCNLab) at the [University of Colorado Boulder](http://www.colorado.edu/) and the [Mnemosyne Project-Team]() at [Inria Bordeaux, France](https://www.inria.fr/en/centre/bordeaux).


## Status & Roadmap

This is a work in progress. Most of the basic algorithms of Leabra are implemented, but some mechanisms are
still missing. While the current implementation passes several quantitative tests of equivalence with
the emergent implementation (8.1.1, r11060), the number and diversity of tests is too low to guarantee that
the implementation is correct yet.

- [x] Unit, Layer, Connection, Network class
- [x] XCAL learning rule
- [x] Basic notebook examples
- [x] Quantitative equivalence with emergent
- [x] Neuron tutorial notebook
- [ ] Inhibition tutorial notebook
- [ ] Weight balance mechanism


## Installation & Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Then, launch Jupyter to see usage examples:
```bash
jupyter notebook index.ipynb
```


## Run Notebooks Online

[Notebooks can be run online](https://beta.mybinder.org/v2/gh/benureau/leabra/master) without installation with the [Binder](http://mybinder.org) service. The service is still experimental, and may be down or unstable.


## Useful Resources

  * [Leabra description](https://grey.colorado.edu/emergent/index.php/Leabra)
  * [emergent homepage](https://grey.colorado.edu/emergent/index.php/Main_Page)
  * [CCNLab homepage](https://grey.colorado.edu/CompCogNeuro/index.php/CCNLab)

## License

To be decided.
