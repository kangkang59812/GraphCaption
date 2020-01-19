#!/bin/sh
id="mul-gcn1-19"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train3.py --id $id \
    --use_gcn True \
    --learning_rate 2e-4 \
    --beam_size 1 \
    --save_checkpoint_every 12000 \
    --batch_size 8 \
    --num_worker 4 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --scheduled_sampling_max_prob 0.5 \
    --checkpoint_path log/log_$id  \
    $start_from \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 4 \
    --use_box 0 \
    --val_images_use -1 \
    --max_epochs 40

python train3.py --id $id \
    --use_gcn True \
    --batch_size 8 \
    --beam_size 1 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --save_checkpoint_every 12000 \
    --start_from log/log_$id \
    --checkpoint_path log/log_$id"_rl" \
    --learning_rate 2e-5 \
    --max_epochs 50 \
    --self_critical_after 0 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --reduce_on_plateau \
    --num_worker 4 \
    --use_box 0 \
    --val_images_use -1 \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3