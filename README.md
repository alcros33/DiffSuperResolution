# Super-Resolution using Diffusion Models

This repository contains the implementation and pre-trained weights for my master's thesis *Exploring the usage of diffusion models for image super-resolution*. It contains implementation and trained weights for the ResShift method (Yue, Wang and Loy, 2023) using both UNet and Difusion Transformer architectures.

## Installation
First follow torch installation instructions [here](https://pytorch.org/get-started/locally/) to correctly install torch according to your cuda version. Used versions are included in `requirements_torch.txt`. Then, install the rest of the requirements.
```
pip install -r requirements.txt
```

## Inference
First download the checkpoints from the releases tab
Place the LR images in a folder, lets call it `LR`.
```
python inference.py checkpoint.pth [unet | transformer] --bs 8 -i LR visible -o results --cuda 
```

## Training
Multiple configuration files are provided in the [config](config) folder, adjust accordingly.
You will need to make 3 files containing the filenames of the train,valid and tests splits relative to the `data_dir` in the config file. A simple way of making them asuming you have them split in 3 folders inside `data_dir` is by using `find train/ -type f > train_list.txt`. 
```
python diffusion_resshift_pixart.py fit --config my_config.yaml
```
or 
```
python diffusion_resshift_unet.py fit --config my_config.yaml
```
In order to finetune from a provided checkpoint you can enter the path to checkpoint file in the ckpt_path field in the config file. 

## License

Distributed under the MIT License. See `LICENSE` for more information.