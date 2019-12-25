#!/bin/sh

python train3.py --id gcn \
--use_gcn True \
--learning_rate 5e-4 \
--self_critical_after 30 \
--beam_size 1 \
--save_checkpoint_every 1800 \
--checkpoint_path savegcn \
--batch_size 64 \
--num_worker 4