
# Introduction
Official codes of paper **_H2C:Hippocampal Circuit-inspired Contiual Learning for Lifelong Trajectory Prediction in Autonomous Driving_**, which is submitted to **_IEEE Transactions on Intelligent Transportation Systems（IEEE T-ITS）_**.

# Dataset
## Original Dataset
The experiments in this study are based on [INTERACTION dataset](https://interaction-dataset.com/).
## Processed Data
- The processed data is available in [Google Drive](https://drive.google.com/drive/folders/1roEeNQJFz777DbPEMf21R3j2BQdRKecp?usp=drive_link).
- Please download the processed data in the direction ``` ./cl_dataset/``` for running the experiments.

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
1. Before running codes, please revise ```root_dir``` in ```_./utils/args_loading.py_``` to your local paths.
2. Parameters for the networks can be also revised in ```_./utils/args_loading.py_```.


## Key Parameters for running the experiments
1. ```--model```: the method you want to train and test. 
2. ```--buffer_size```: the memory size of the continual learning methods to run, and set as 0 when using the vanilla method.
3. ```--dataset```: set as "seq-interaction" when continual training, set as "joint-interaction" when joint training.
4. ```--train_task_num```: the number of tasks in continual training.
5. ```--debug_mode```: _True_ or _1_ when you are debugging, only a few batches of samples will be used in each task for a convenient check. _False_ or _0_ in the formal training.  
6. ```--num_tasks```: the number of continual tasks for testing.


# Codes
## File Structure
```text
├── README.md
├── cl_data_stream
│   ├── joint_dataset.py
│   ├── seq_dataset.py
│   └── traj_dataset.py
├── cl_model
│   ├── __init__.py
│   ├── agem.py
│   ├── continual_model.py
│   ├── der.py
│   ├── h2c.py
│   ├── gss.py
│   └── vanilla.py
├── experiments
│   ├── joint_training.py
│   ├── seq_training_all_task.py
│   └── testing_1_task.py
├── test_CL.py
├── test_joint.py
├── train_CL.py
├── train_joint.py
├── traj_predictor
│   ├── __init__.py
│   ├── baselayers.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── interaction_model.py
│   ├── losses.py
│   ├── traj_para.py
│   └── utils.py
├── results
│   ├── logs
│   └── weights
├── utils
│   ├── args_loading.py
│   ├── metrics.py
│   ├── completion_buffer.py
│   └── separation_buffer.py
└── visualization_utils
    ├── dict_utils.py
    ├── dictionary.py
    ├── extract_original_tv_info.py
    └── map_vis_without_lanelet.py
```
- Main functions for training and testing are the python scripts in the root direction.
- The folder ```cl_model``` contains the proposed H2C and compared CL baselines in experiments.
- The folder ```traj_predictor``` includes the files to construct the trajectory prediction model.
- Scripts for data pre-processing are provided in the folder ```cl_datastream```.

## Training and Testing via the Bash File
After adding the Executable Permissions to the provided bash file (_bash_training_and_test.sh_), you can directly run the training and testing with command:
```
./bash_training_and_test.sh
```

