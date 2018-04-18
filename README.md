# DEEP QC
Code for the paper Vladimir S. Fonov, Mahsa Dadar, The PREVENT-AD Research Group, D. Louis Collins **"Deep learning of quality control for stereotaxic registration of human brain MRI"** 

## Dependencies:
* Torch:`display xlua cudnn optim paths`, optionally: `minc2-simple` 
* PyTorch: `scikit-image tensorboard tensorboardX `, optionally : `minc2-simple`
* minc2-simple (optional): https://github.com/vfonov/minc2-simple

## Files:
* Shell scripts:
    * `download_minimal_results.sh`  - download pretrained model to run automatic qc
    * `make_figures.sh`  - Draw Figure 2 and 3 for the paper
* Torch implementation, directory `lua`:
    * `run_all_cases.sh` - master script to run all experiments mentioned in the paper
    * `aqc_training.lua` - LUA script to run individual training/testing experiemnt, requires GPU with CUDA
    * `aqc_apply.lua` - LUA script to apply pre-trained model to existing data (Run automatic QC), runs on CPU by default
    * `aqc_data.lua` - internal LUA module implementing data loading routines
    * `aqc_model.lua` - internal LUA module implementing neural net 
* PyTorch implementation, directory `python`:
    * `run_all_experiments.sh` - run experiments with different versions of ResNet and SquezeNet
    * `aqc_apply.py` - apply pre-trained network
    * `aqc_convert_to_cpu.py`- helper script to convert network from GPU to CPU 
    * `aqc_data.py` - module to load QC data
    * `aqc_training.py` - deep nearal net training script
    * `model/resnet_qc.py` - module with ResNET implementation, based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    * `model/squezenet_qc.py` - module with SqueezeNet implementation, based on https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
    * `model/util.py` - various helper functions
* Image files:
    * `mni_icbm152_t1_tal_nlin_sym_09c_0.jpg`,`mni_icbm152_t1_tal_nlin_sym_09c_1.jpg`,`mni_icbm152_t1_tal_nlin_sym_09c_2.jpg` - reference slices, needed for both training and running pretrained model
* `results` - directory with outputs, containes pre-trained models
* `data` - RAW and intermediate datafiles will be placed are here
* R scripts:
  * `aqc_analysis.R` - Draw Figure 5,6,7
  * `aqc_analysis_one_long.R`  - Draw Figure 4
  * `aqc_analysis_r18_ref.R`  - Draw Figure 8
  * `summary.R`  - calculate summary stats 
  * `multiplot.R` - internal module for making stacked plots in ggplot

## Running:
* `./run_all_cases.sh` - will run all the cases (needs Nvidia Titan-X), using torch implementation
* in `python` directory `run_all_experiments.sh` - will try to train all networks
* Torch (lua) version: `th aqc_apply.lua -volume <input.mnc>` or `th aqc_apply.lua -image <images_base>` apply pre-trained model to either a 3D minc volume or set of three images 
* PyTorch (python) version: `python3 python/aqc_apply.py --volume <input.mnc>` or `python3 python/aqc_apply.py --image <image base>`

## Validate correct operation (requires minc-toolkit and minc2_simple python module)
```
# create a file with 30 degree rotation transform
param2xfm -rotations 30 0 0  rot_30.xfm
# apply to a template:
itk_resample /opt/minc/share/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc --transform rot_30.xfm bad.mnc

# run QC script on good scan
# should print "Pass"
python3 python/aqc_apply.py --volume /opt/minc/share/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc

# now on "bad"
# should print "Fail"
python3 python/aqc_apply.py --volume bad.mnc
```
