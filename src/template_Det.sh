#!/bin/bash
#=========================================================
# This is a script for submitting multiple jobs using parameters.
# The example is taken from https://www.osc.edu/resources/getting_started/howto/howto_submit_multiple_jobs_using_parameters
# This script should be used as a template alongside
# with a CSV file with parameters definition and a Python script for automatic job submission
#=========================================================

#SBATCH -n 1
#SBATCH -s
#SBATCH --exclude=hilbert
##SBATCH --nodelist sobolev
##SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically first among nodes and then among sockets within a node
#SBATCH --time=72:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH -e slurm-%j_cf.log           # Path to the standard output and error files relative to the working directory
#SBATCH -o slurm-%j_cf.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$tatiana.medvedeva@skoltech.ru

# Print the time and date at the beginning
echo "Start date:" `date`
echo "HOST:" `hostname`
#echo $PBS_O_WORKDIR
echo "JOB NAME: $PBS_JOBNAME"


# Define interpretator and solver
python=python3
solver=~/Detonation/periodicFric/DetPeriodicFriction.py


# Creating directory and run simulation there
#dir="${OUT_DIR}/E${ENERGY}L${LENGHT}N${NODES}cf${CF}_from${START}to${FINISH}"
dir="${OUT_DIR}/E${ENERGY}L${LENGHT}N${NODES}cf${CF}kf${START}"
mkdir -p ${dir}
cd ${dir}
$python $solver ${ENERGY} ${LENGHT} ${NODES} ${CF} ${EPS} ${START} ${FINISH} ${STEP}

# Print the time and date at the end
echo "End date:" `date`
