
# Manual on Evaluation


## Environment Setup   

1. Operating system

    Ubuntu 18.04 

2. Python version

    python = 3.7

3. Packages installed in different conda environment

   1. [base](requirements/base_env_installed_packages.md)
   2. [paddlepaddle](requirements/paddlepaddle_env_installed_packages.md)
   3. [pytorch](requirements/pytorch_env_installed_packages.md)
   4. [tensorflow](requirements/tensorflow_env_installed_packages.md)



## Submitted Files

When participants submit their developed code and model in a compressed file, the extracted items should look like as follows: 

```
    ./kddcup22-sdwpf-evaluation     (required)
    | --- __init__.py         
    | --- [base/paddle/pytorch/tensorflow]
         | --- __init__.py
         | --- evaluation.py  (required)
         | --- preeict.py     (required)
         | --- metrics.py     (required)
         | --- test_data.py   (required)
         | --- ...
   | --- model_subfolder
         | --- ... 
   | --- ...
```

The extracted folder is named as 'kddcup22-sdwpf-evaluation'. 
This folder should contain a sub-folder named as 'base'/'paddlepaddle'/'pytorch'/'tensorflow', which depends on the machine learning framework one adopts. 
In the code sub-folder, evaluation.py, predict.py, metrics.py and test_data.py are required.
And, the evaluation.py, metrics.py and test_data.py should be the released version, which will be checked with MD5 when the submission starts.
In the predict.py script, the forecast interface should be implemented by the participants, and the forecast interface takes a dictionary that consists of a number of settings as the parameter. 


### Shell script for running the evaluation (TBD)


The following shell script illustrates how we initiate the evaluation procedure, 
which will be determined after discussing with the AIStudio engineers.

```
    #!/usr/bin/env bash

   if [ $# -ne 1 ]; then
      echo "The machine learning framework (i.e. base/paddlepaddle/pytorch/tensorflow) is missing, which MUST be provided"
      echo "For example, sh run_evaluation.sh paddlepaddle"
      exit 1
   fi
   
   ML_framework=$1
   data_path="path/to/data"
   filename="sdwpf_baidukddcup2022_full.csv"
   path_to_test_x="path/to/data/sdwpf_baidukddcup2022_test/test_x"
   path_to_test_y="path/to/data/sdwpf_baidukddcup2022_test/test_y"
   predict_file="predict.py"
   is_debug=True
   
   conda activate $1
   python "./kddcup22-sdwpf-evaluation/"$ML_framework"/evaluation.py" \
       --data_path $data_path \
       --filename $filename \
       --path_to_test_x $path_to_test_x \
       --path_to_test_y $path_to_test_y \
       --pred_file "./kddcup22-sdwpf-evaluation/"$ML_framework"/"$predict_file \
       --is_debug $is_debug \
       --framework $ML_framework
```

As can be seen in the above shell script, several arguments will be specified.
In other words, in the script handling the parameters and settings, e.g. the prepare.py script in the baseline code, 
the participants should maintain the arguments mentioned in the above shell script. 
