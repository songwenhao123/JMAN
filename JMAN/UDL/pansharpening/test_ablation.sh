#!/bin/bash

# Function to test pansharpening performance with specified arguments
run_test_pansharpening() {
    mode="$1"
    dataset="$2"
    layers="$3"
    resgroups="$4"
    fusionblocks="$5"
    resblocks="$6"
    n_colors="$7"
    act_variant="$8"
    num_repeats="$9"
    out_dir="./results/ACT_24Aug_e400_bs48_rep2_rb3"

    python -u run_test_pansharpening.py \
        --arch LANet \
        --val_set "test_$dataset" \
        --test_mode "$mode" \
        --eval \
        --data_dir "$DATA_DIR" \
        --n_layers "$layers" \
        --n_resgroups "$resgroups" \
        --n_fusionblocks "$fusionblocks" \
        --n_resblocks "$resblocks" \
        --n_colors "$n_colors" \
        --act_variant "$act_variant"\
        --num_repeats "$num_repeats" \
        --resume_from "${out_dir}/LANet/${act_variant}/${dataset}/best.pth.tar"
}

# Check if running inside an Apptainer container
if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/usr/bin/python
    # DATA_DIR=/home/qilei/Experiments/PanCollection/01-DL-toolbox/UDL/Data/pansharpening
    if [ "$HOSTNAME" == "*sbg*" ]; then
        DATA_DIR=/data/home/acw565/Datasets/Multimodal/pancollection
    else
        # DATA_DIR=/home/qilei/Experiments/PanCollection/01-DL-toolbox/UDL/Data/pansharpening
        DATA_DIR=/scratch/pancollection
    fi
    KNOCKKNOCK_PATH="/usr/local/bin/knockknock"
    echo "Running inside an Apptainer container."
else
    PYTHON=$HOME/Installs/conda/envs/pans/bin/python
    KNOCKKNOCK_PATH=$HOME/Installs/conda/envs/pans/bin/knockknock
    DATA_DIR=/scratch/pancollection
    # DATA_DIR=/data/home/acw565/Datasets/Multimodal/pancollection
    echo "Not running inside an Apptainer container."
fi

NOTIFY_COMMAND=$(eval echo "$KNOCKKNOCK_PATH teams --webhook-url 'https://qmulprod.webhook.office.com/webhookb2/57a949a2-37f2-4edc-8351-c036a83f01bb@569df091-b013-40e3-86ee-bd9cb9e25814/IncomingWebhook/b0b850be28734d0db81eda141cdbd688/d478ee1b-5c8f-42c2-b41a-f4d999173fef'")

# Example usage:
run_test_pansharpening "downsample" "gf2" "2" "1" "1" "3" "4" "lwln_base" "2"&
# run_test_pansharpening "downsample" "qb" "2" "1" "1" "3" "4" "lwln_base" "2"&
# run_test_pansharpening "downsample" "wv3" "2" "1" "1" "3" "8" "lwln_base" "2"&


# run_test_pansharpening "downsample" "qb"  "2" "1" "1" "3" "4" "no_interaction" "2"&
# run_test_pansharpening "downsample" "qb" "2" "1" "1" "3" "4" "inter_interaction" "2"&
# run_test_pansharpening "downsample" "qb" "2" "1" "1" "3" "4" "intra_interaction" "2"&


# run_test_pansharpening "downsample" "gf2"  "2" "1" "1" "3" "4" "lwln_swap" "4"
# run_test_pansharpening "downsample" "gf2" "2" "1" "1" "3" "4" "lwln_contrastswap" "4"
wait

# apptainer exec --nv $HOME/Installs/containers/pytorch_23.06_pans.sif bash -c "cd $HOME/Experiments/PanCollection/01-DL-toolbox/UDL/pansharpening && bash test.sh"
