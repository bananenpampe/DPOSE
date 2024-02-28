#!/bin/bash

# copies the utilities for the BPNN-SOAP model in this directory
git clone -b move-rascaline git@github.com:bananenpampe/H2O.git

# Install cargo v<1.75.0
mkdir -p "$PWD/rust_installation"

#already fixed in newer rascaline versions
export MACOSX_DEPLOYMENT_TARGET=11

# Set the installation directories
export RUSTUP_HOME="$PWD/rust_installation/rustup"
export CARGO_HOME="$PWD/rust_installation/cargo"

# Download and install rustup, and thus Rust, into the specified directories
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Update the current shell session
source "${CARGO_HOME}/env"

echo "Rust has been installed in $PWD/rust_installation"
rustc --version

rustup install 1.74.0

# Set Rust 1.7.4 as the default version for compatibility
rustup default 1.74.0

# Verify the installation
rustc --version


# pip cache purge
# install cpu pytorch 2.0.1

pip install cmake numpy
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.0.1

export RASCALINE_COMMIT=55ad1f14363812639c865801a6d517d8c5f363f6
export METATENSOR_OPERATIONS_COMMIT=646f1f49c1d9cc18c3fa0d62e2c15e630b8d9d8c
export METATENSOR_COMMIT=2248a3c1feb440a32922ffac98a0592474b48f2d

export PIP_FLAGS="--no-deps --no-build-isolation --check-build-dependencies"

pip install $PIP_FLAGS "metatensor @ git+https://github.com/lab-cosmo/metatensor@$METATENSOR_COMMIT"
pip install $PIP_FLAGS "metatensor-core @ git+https://github.com/lab-cosmo/metatensor@$METATENSOR_COMMIT#subdirectory=python/metatensor-core"
pip install $PIP_FLAGS "metatensor-torch @ git+https://github.com/lab-cosmo/metatensor@$METATENSOR_COMMIT#subdirectory=python/metatensor-torch"

pip install $PIP_FLAGS "metatensor-operations @ git+https://github.com/lab-cosmo/metatensor@$METATENSOR_OPERATIONS_COMMIT#subdirectory=python/metatensor-operations"

pip install $PIP_FLAGS "rascaline @ git+https://github.com/luthaf/rascaline@$RASCALINE_COMMIT"
pip install $PIP_FLAGS "rascaline-torch @ git+https://github.com/luthaf/rascaline@$RASCALINE_COMMIT#subdirectory=python/rascaline-torch"

pip install -r requirements.txt
