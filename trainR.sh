#!/bin/sh
id="gcn-96000"

python train3.py --id $id \
--use_gcn True \
--batch_size 32 \
--beam_size 1 \
--input_encoding_size 1024 \
--rnn_size 1024 \
--save_checkpoint_every 4000 \
--start_from log/log_gcn \
--checkpoint_path log/log_gcn_rl \
--learning_rate 2e-5 \
--max_epochs 40 \
--self_critical_after 0 \
--learning_rate_decay_start -1 \
--scheduled_sampling_start -1 \
--reduce_on_plateau \
--num_worker 4 \
--use_box 0