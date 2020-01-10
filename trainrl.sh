#!/bin/sh
id="gcn"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train3.py --id $id \
    --use_gcn True \
    --learning_rate 4e-4 \
    --beam_size 1 \
    --save_checkpoint_every 4000 \
    --batch_size 32 \
    --num_worker 4 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --label_smoothing 0.2 \
    --scheduled_sampling_max_prob 0.5 \
    --checkpoint_path log/log_$id  \
    $start_from \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 4 \
    --use_box 0

python train3.py --id $id \
    --use_gcn True \
    --batch_size 32 \
    --beam_size 1 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --save_checkpoint_every 4000 \
    --start_from log/log_$id \
    --checkpoint_path log/log_$id"_rl" \
    --learning_rate 2e-5 \
    --max_epochs 40 \
    --self_critical_after 0 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --reduce_on_plateau \
    --num_worker 4 \
    --use_box 0