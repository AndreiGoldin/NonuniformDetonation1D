#!/bin/bash
# E_list=(25)
# wns_list=(1)
# amps_list=0.1
E_list=(25 26 27.5)
wns_list=(0.05 0.08 0.1 0.15 0.2 0.4 0.8 1 1.5 2)
amps_list=$(seq -0.5 0.1 0.5)
for amp in $amps_list; do
    for wn in "${wns_list[@]}"; do
        for E in "${E_list[@]}"; do
            sbatch calc_script.sh $E $amp $wn
            # echo $E $amp $wn;
        done;
    done;
done

