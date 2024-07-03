#!/bin/bash

pip install --upgrade fcd
fname=$(python -c "from guacamol import frechet_benchmark; print(frechet_benchmark.__file__)")
sed -i 's/ChemNet_v0.13_pretrained.h5/ChemNet_v0.13_pretrained.pt/g' $fname