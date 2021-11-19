# DEEP QC

Code for the paper Vladimir S. Fonov, Mahsa Dadar, The PREVENT-AD Research Group, D. Louis Collins **"DARQ: Deep learning of quality control for stereotaxic registration of human brain MRI"**.

*Updated version of the previosly available ["Deep learning of quality control for stereotaxic registration of human brain MRI"](https://doi.org/10.1101/303487)*

## Installation (Python version) using *conda* for inference

* CPU version
    ```
    conda install pytorch-cpu==1.7.1 torchvision==0.8.2 cpuonly -c pytorch 
    conda install scikit-image
    ```
* GPU version
    ```
    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=<your cuda version>  -c pytorch 
    conda install scikit-image
    ```
* (optional) minc toolkit and minc2-simple
   ```
   conda install -c vfonov minc-toolkit-v2 minc2-simple
   ```

## Running

* Inference python (pytorch) version: `python3 python/aqc_apply.py --volume <input.mnc>` or `python3 python/aqc_apply.py --image <image base>`
* Inference lua (torch) version: `th aqc_apply.lua -volume <input.mnc>` or `th aqc_apply.lua -image <images_base>` apply pre-trained model to either a 3D minc volume or set of three images 
* Training python version in `python` directory `run_all_experiments.sh` - will try to train all networks
* Training lua (torch) version: `./run_all_cases.sh` - will run all the cases (needs Nvidia Titan-X)

## Dependencies

* trainig dependencies: `scikit-image tensorboard`, optionally : `minc2-simple`
* minc2-simple (optional): https://github.com/vfonov/minc2-simple

## Files

* Shell scripts:
    * `download_minimal_models.sh`  - download QCResNET-18 with reference pretrained model to run automatic qc (43mb)
    * `download_all_models.sh`  - download all pretrained models to run automatic qc 
* Directory `python`:
    * `run_all_experiments.sh` - run experiments with different versions of ResNet and SquezeNet
    * `aqc_apply.py` - apply pre-trained network
    * `aqc_convert_to_cpu.py`- helper script to convert network from GPU to CPU
    * `aqc_data.py` - module to load QC data
    * `aqc_training.py` - deep nearal net training script
    * `model/resnet_qc.py` - module with ResNET implementation, based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    * `model/util.py` - various helper functions
    * `*.R` - R scripts to generete figures for the paper
* Image files:
    * `mni_icbm152_t1_tal_nlin_sym_09c_0.jpg`,`mni_icbm152_t1_tal_nlin_sym_09c_1.jpg`,`mni_icbm152_t1_tal_nlin_sym_09c_2.jpg` - reference slices, needed for both training and running pretrained model
* `results` - figures for the paper
* `data` - reference images

## Validating correct operation (requires minc-toolkit and minc2_simple python module)

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
