#!/bin/sh

echo "TRAVIS_OS_NAME = $TRAVIS_OS_NAME"
echo "TRAVIS_PYTHON_VERSION = $TRAVIS_PYTHON_VERSION"
echo "TOXENV = $TOXENV"

if [ $TRAVIS_OS_NAME = 'osx' ]; then

  # Install miniconda on macOS
  curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
  mv Miniconda3-latest-MacOSX-x86_64.sh miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  . "$HOME/miniconda/etc/profile.d/conda.sh"
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
  . "$HOME/miniconda/etc/profile.d/conda.sh"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda

  # Useful for debugging any issues with conda
  conda info -a
fi

case "${TOXENV}" in
py36)
  # Install Mac Python3.6 environment
  conda env create -f environment-36.yml
  ;;
py37)
  # Install some custom Python 3.3 requirements on macOS
  conda env create -f environment.yml
  ;;
esac
