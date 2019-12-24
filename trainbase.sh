#!/bin/sh

python train3.py --id fc_rl \
--use_gcn False \
--start_from save_rl \
--checkpoint_path save_rl 
--learning_rate 5e-5 \
--self_critical_after 30 \
--cached_tokens coco-train-idxs \
--beam_size 1