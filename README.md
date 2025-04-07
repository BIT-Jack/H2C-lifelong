
# Introduction
Official codes of paper **_H2C:Hippocampal Circuit-inspired Contiual Learning for Lifelong Trajectory Prediction in Autonomous Driving_**, which is submitted to _IEEE Transactions on Pattern Analysis and Machine Intelligence （IEEE T-PAMI）_.

# Processed Data
The experiments in this work are based on [INTERACTION dataset](https://interaction-dataset.com/).
The processed data is available in this link for [download](https://drive.google.com/drive/folders/1roEeNQJFz777DbPEMf21R3j2BQdRKecp?usp=drive_link).

# Implementations
## Enviroment
1. System: The codes can be run in **Ubuntu 22.04 LTS**.
2. **Python = 3.9**
3. **Pytorch = 2.0.0**
4. Other required packages are provided in "**requirements.txt**":
```
 pip install -r requirements.txt
```
## Configurations
1. Before running codes, please revise "**root_dir**" in "_./utils/conf.py_" to your local paths.
2. Parameters for the networks can be also revised in "_./utils/config.py_".


## Key Parameters for running the experiments
1. **--model**: the method you want to train and test. 
2. **--buffer_size**: the memory size of the continual learning methods to run, and set as 0 when using the vanilla method.
3. **--dataset**: set as "seq-interaction" when continual training, set as "joint-interaction" when joint training.
4. **--train_task_num**: the number of tasks in continual training.
5. **--debug_mode**: _True_ or _1_ when you are debugging, only a few batches of samples will be used in each task for a convenient check. _False_ or _0_ in the formal training.  
6. **--num_tasks**: the number of continual tasks for testing.


# Running
## Simple usage of the bash file
After adding the Executable Permissions to the provided bash file (_bash_training_and_test.sh_), you can directly run the training and testing with command:
```
./bash_training_and_test.sh
```