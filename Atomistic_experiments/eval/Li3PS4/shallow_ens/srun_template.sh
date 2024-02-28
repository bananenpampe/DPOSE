#!/bin/bash

#SBATCH --job-name=water_bpnn
#SBATCH --mail-type=FAIL

#SBATCH --output=log.out
#SBATCH --ntasks-per-node=72
#SBATCH --mem=480GB
#SBATCH --time=02:00:00
#SBATCH --nodes=1
##SBATCH --qos=serial
#SBATCH --get-user-env

set -e


python3 -u training.py 

