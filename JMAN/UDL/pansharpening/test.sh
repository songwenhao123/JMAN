#!/bin/bash

# Function to test pansharpening performance with specified arguments
run_test_pansharpening() {
    mode="$1"
    dataset="$2"
    arch="$3"
    out_dir="./results"

    python -u run_test_pansharpening.py \
        --arch "$arch" \
        --val_set "test_$dataset" \
        --test_mode "$mode" \
        --eval \
        --data_dir "$DATA_DIR" \
        --resume_from "${out_dir}/${arch}/${arch}/${dataset}/best.pth.tar"
}

# Check if running inside an Apptainer container
if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/usr/bin/python
    # DATA_DIR=/data/home/acw565/Datasets/Multimodal/pancollection
    # DATA_DIR=/home/qilei/Experiments/PanCollection/01-DL-toolbox/UDL/Data/pansharpening
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

# Example usage:
run_test_pansharpening "fullsample" "gf2" "Panformer" &
run_test_pansharpening "fullsample" "qb"  "Panformer" &
run_test_pansharpening "fullsample" "wv3" "Panformer" &
wait

# apptainer exec --nv $HOME/Installs/containers/pytorch_23.06_pans.sif bash -c "cd $HOME/Experiments/PanCollection/01-DL-toolbox/UDL/pansharpening && bash test.sh"
