#!/bin/bash

# Function to run pansharpening with specified arguments
run_pansharpening() {
    gpu_device="$1"
    dataset="$2"
    layers="$3"
    resgroups="$4"
    fusionblocks="$5"
    resblocks="$6"
    n_colors="$7"
    act_variant="$8"
    num_repeats="$9"

    CUDA_VISIBLE_DEVICES="$gpu_device" \
        ${NOTIFY_COMMAND} \
        $PYTHON -u run_pansharpening.py \
        --arch LANet \
        --act_variant "$act_variant" \
        --train_set "$dataset" \
        --val_set "valid_$dataset" \
        --epochs 401 \
        --seed 821 \
        --data_dir "$DATA_DIR" \
        --out_dir "./results/ACT_27Aug_e400_bs48_rep${num_repeats}_rb${resblocks}" \
        --samples_per_gpu 48 \
        --n_layers "$layers" \
        --n_resgroups "$resgroups" \
        --n_fusionblocks "$fusionblocks" \
        --n_resblocks "$resblocks" \
        --n_colors "$n_colors" \
        --num_repeats "$num_repeats" \
        --start_val "100" &
}

# Check if running inside an Apptainer container
if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/usr/bin/python
    # DATA_DIR=/data/home/acw565/Datasets/Multimodal/pancollection
    # DATA_DIR=/home/qilei/Experiments/PanCollection/01-DL-toolbox/UDL/Data/pansharpening
    # Check hostname and set DATA_DIR accordingly
    if [ "$HOSTNAME" == "*sbg*" ]; then
        DATA_DIR=/data/home/acw565/Datasets/Multimodal/pancollection
    else
        DATA_DIR=/scratch/pancollection
    fi
    KNOCKKNOCK_PATH="/usr/local/bin/knockknock"
    echo "Running inside an Apptainer container."
else
    PYTHON=$HOME/Installs/conda/envs/lavis/bin/python
    KNOCKKNOCK_PATH="$HOME/Installs/conda/envs/dgm4/bin/knockknock"
    DATA_DIR=/scratch/pancollection
    echo "Not running inside an Apptainer container."
fi

NOTIFY_COMMAND=$(eval echo "$KNOCKKNOCK_PATH teams --webhook-url 'https://qmulprod.webhook.office.com/webhookb2/57a949a2-37f2-4edc-8351-c036a83f01bb@569df091-b013-40e3-86ee-bd9cb9e25814/IncomingWebhook/b0b850be28734d0db81eda141cdbd688/d478ee1b-5c8f-42c2-b41a-f4d999173fef'")

# Run the pansharpening with different datasets and configurations in parallel
# run_pansharpening "0" "wv3" "2" "1" "1" "3" "8" "lwln_base" "4"
# run_pansharpening "0" "wv3" "2" "1" "1" "3" "8" "no_interaction" "4"
# run_pansharpening "1" "wv3" "2" "1" "1" "3" "8" "inter_interaction" "4"
run_pansharpening "1" "wv3" "2" "1" "1" "3" "8" "intra_interaction" "4"


# run_pansharpening "1" "gf2" "2" "1" "1" "3" "4" "lwln_swap" "2"
# run_pansharpening "0" "gf2" "2" "1" "1" "3" "4" "lwln_contrastswap" "2"

wait



# apptainer exec --nv $TMPDIR/pytorch_23.06_pans.sif bash -c "cd $HOME/Experiments/PanCollection/01-DL-toolbox/UDL/pansharpening && bash train.sh"
