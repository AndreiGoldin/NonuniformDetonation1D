#!/bin/bash
#=========================================================
# This is a script for submitting multiple jobs using parameters.
# The example is taken from https://www.osc.edu/resources/getting_started/howto/howto_submit_multiple_jobs_using_parameters
# This script should be used as a template alongside
# with a CSV file with parameters definition and a Python script for automatic job submission
#=========================================================
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00
#PBS -j oe
# PBS -o ${HOME}/Detonation/periodicFric/out_err.$PBS_JOBNAME

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
dir="${OUT_DIR}/E${ENERGY}L${LENGHT}N${NODES}cf${CF}"
mkdir -p ${dir}
cd ${dir}
$python $solver ${ENERGY} ${LENGHT} ${NODES} ${CF} ${EPS} ${START} ${FINISH} ${STEP}

# Print the time and date at the end
echo "End date:" `date`
