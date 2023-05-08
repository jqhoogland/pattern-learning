#!/bin/bash

while true
do
    echo "Starting W&B sweep agent..."
    wandb agent --project grokking jqhoogland/grokking/$SWEEP_ID
    
    echo "W&B sweep agent stopped. Restarting in 10 seconds..."

    sleep 10
done