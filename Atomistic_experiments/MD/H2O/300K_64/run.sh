#!/bin/bash

which python
rm -f /tmp/ipi_alchem_H2O_5

i-pi simulation.xml &

sleep 10s

echo "Current time: $(date +"%H:%M:%S")"

i-pi-py_driver -m lightning -a alchem_H2O_5 -u -o ../../../eval/H2O/shallow_ens/example.ckpt,start_64.xyz


