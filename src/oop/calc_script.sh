#!/bin/bash
#SBATCH -n 1
#SBATCH --time=12:00:00

solver='/trinity/home/andrei.goldin/NonuniformDetonation1D/src/oop/one_halfwave.py'
E=$1
amp=$2
wn=$3
python3 $solver $E $amp $wn

