#!/bin/bash

echo "TRAVIS_OS_NAME = $TRAVIS_OS_NAME"
echo "TRAVIS_PYTHON_VERSION = $TRAVIS_PYTHON_VERSION"
echo "TOXENV = $TOXENV"

if [ $TRAVIS_OS_NAME = 'osx' ]; then

  # Install miniconda on macOS
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  source "$HOME/miniconda/etc/profile.d/conda.sh"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda

  # Useful for debugging any issues with conda
  conda info -a

else

  # Install miniconda on Linux
  sudo apt-get update
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  source "$HOME/miniconda/etc/profile.d/conda.sh"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda

  # Useful for debugging any issues with conda
  conda info -a
fi
