# CVD Risk Distribution Shift
This project mainly focus on the validation the impact of distribution shift on the risk prediction of major CVD event, here are the brief instructions below
- requirement.txt includes Python packages to run this project
- The folder "Jupyer" includes the [training script for CPH models](https://github.com/deepmedicine/CVDRiskDistributionShift/blob/main/jupyer/CPH.ipynb)
- the [yaml file](https://github.com/deepmedicine/CVDRiskDistributionShift/tree/main/example_config) in the example_config folder provides necessary parameters to train the BEHRT model
- when the yaml file is ready, run the main.py to train the BEHRT model, save_path is the directory to save log and model checkpoint
-- python /main.py --params exmaple_config/bert.yaml --save_path ./

