#!/bin/sh

cp -r $1 $2
cd $2
mv infos_fc-best.pkl infos_fc_rl-best.pkl 
mv infos_fc.pkl infos_fc_rl.pkl 
cd ../