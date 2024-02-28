#!/bin/bash

#SBATCH --job-name=water_bpnn
#SBATCH --mail-type=FAIL

#SBATCH --output=log.out
#SBATCH --ntasks-per-node=36
#SBATCH --mem=490GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=bigmem
##SBATCH --qos=serial
#SBATCH --get-user-env

set -e


python3 -u training.py 

