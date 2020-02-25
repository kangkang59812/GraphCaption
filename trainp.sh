#!/bin/sh
id="plstm2-15"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python trainp.py --id $id \
    --learning_rate 5e-4 \
    --beam_size 1 \
    --save_checkpoint_every 3000 \
    --batch_size 40 \
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
    --learning_rate_decay_every 3 \
    --val_images_use -1 \
    --max_epochs 50 \
    --learning_rate_decay_rate 0.8

python trainp.py --id $id \
    --batch_size 40 \
    --beam_size 1 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --save_checkpoint_every 3000 \
    --start_from log/log_$id \
    --checkpoint_path log/log_$id"_rl" \
    --learning_rate 2e-5 \
    --max_epochs 70 \
    --self_critical_after 0 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --reduce_on_plateau \
    --num_worker 4 \
    --val_images_use -1
    