#!/bin/bash

# Store the current working directory
current_dir=$(pwd)

# Change to the directory containing the MATLAB script
cd ${HOME}/Experiments/PanCollection/02-Test-toolbox-for-traditional-and-DL
# Run MATLAB commands with different arguments, print the output, and save it to log.txt
if [ "$HOSTNAME" == "*sbg*" ]; then
        module load matlab
    else
        export PATH="/scratch/installs/matlab2019/bin:$PATH" 
    fi
# matlab -r "Demo_Reduced_Resolution('GF2', 'gf2', 'lwln_base'); Demo_Reduced_Resolution('GF2', 'gf2', 'lwln_swap'); Demo_Reduced_Resolution('GF2', 'gf2', 'lwln_contrastswap'); quit;" | tee ${current_dir}/log.txt
matlab -r "Demo_Reduced_Resolution('GF2', 'gf2', 'lwln_base'); Demo_Reduced_Resolution('QB', 'qb', 'lwln_base'); Demo_Reduced_Resolution('WV3', 'wv3', 'lwln_base'); quit; " | tee ${current_dir}/log.txt
# Return to the original directory
cd "$current_dir"
