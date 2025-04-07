#!/bin/bash

#training and test CL...
echo " Running Continual Training EXP"
echo $(pwd)
python train_CL.py  --model h2c --buffer_size 1000 --debug_mode 0 & P1=$!
wait $P1
python test_CL.py --model h2c --buffer_size 1000 --num_tasks 8 & P2=$!
wait $P2
echo "H2C(ours) Finished"


python train_CL.py --model vanilla --buffer_size 0 --debug_mode 0 & P3=$!
wait $P3
python test_CL.py --model vanilla --buffer_size 0 & P4=$!
wait $P4
echo "Vanilla (the non-CL baseline) Finished"




echo "All scripts are executed."
 
