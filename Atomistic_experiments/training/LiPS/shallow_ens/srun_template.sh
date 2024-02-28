#!/bin/bash

#SBATCH --job-name=water_bpnn
#SBATCH --mail-type=FAIL

#SBATCH --output=log.out
#SBATCH --ntasks-per-node=72
#SBATCH --mem=490GB
#SBATCH --time=48:00:00
#SBATCH --nodes=1
##SBATCH --qos=serial
#SBATCH --get-user-env

set -e


python3 -u training.py 

