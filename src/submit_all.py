#!/usr/bin/env python
# Script for submitting all the jobs for shock speed or spectra calculation
import csv, subprocess

# out_dir="/home/t.medvedeva/Detonation/cal_for_article
out_dir="/home/a.goldin/Detonation/periodicFric"
parameter_file_full_path = f"{out_dir}/jobs.csv"

with open(parameter_file_full_path, "r") as csvfile:
    # reader = csv.reader(csvfile, delimiter=';')
    reader = csv.reader(csvfile, delimiter=',')
    for job in reader:
        # print(f"mkdir -p {out_dir}")
        # dir_status = subprocess.call(f"mkdir -p {out_dir}}", shell=True)
        first_part = f"sbatch --export=ENERGY={job[0]},LENGHT={job[1]},NODES={job[2]},CF={job[3]},EPS={job[4]},START={job[5]},"
        second_part = f"FINISH={job[6]},STEP={job[7]},OUT_DIR={out_dir}"
        third_part = f" --job-name E{job[0]}L{job[1]}N{job[2]}c_f{job[3]}eps{job[4]}_from{job[5]}to{job[6]}step{job[7]}"
        fourth_part = f" ~/Detonation/periodicFric/template_Det.sh"
        qsub_command = ''.join([first_part, second_part, third_part, fourth_part])
        #qsub_command = """qsub -v TYPE={0},LENGHT={1},NODES={2},AMP={3},START={4},
        #FINISH={5},OUT_DIR={7},ENERGY={6} -N {0}E{6}L{1}N{2}amp{3}_from{4}to{5}
        #~/Detonation/scripts/template_Det.sh""".format(*job, out_dir)

        print(qsub_command) # Uncomment this line when testing to view the qsub command

        # Comment the following 3 lines when testing to prevent jobs from being submitted
        exit_status = subprocess.call(qsub_command, shell=True)
        if exit_status == 1:  # Check to make sure the job submitted
             print("Job {0} failed to submit".format(qsub_command))
print("Done submitting jobs!")


