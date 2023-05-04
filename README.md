# Statistical Characteristics Estimation for Underwater Image Restoration (SCEIR) SPL2023
This is an official pytorch implement of "Atmospheric Scattering Model Induced Statistical Characteristics Estimation
for Underwater Image Restoration".

<img src="comparison4_2.png"/>

## Requirements
We implement the experiments in the following environment.
1. python=3.9.12
2. torch==1.10.2
3. torchvision=0.11.3
4. tqdm

## Running

### Testing
Put the test images into ./test_input or change the option "--test_input" to your image dir

```
python test.py --save_extra --gpu YOUR_DEVICE --test_input YOUR_IMAGE_DIR
```

### Training
Put the pair training images into ./dataset, and changes the option "--patch_low", "--patch_high", 
"--eval_low" and "--eval_high" to the relevant path.

```
python train.py --gpu YOUR_DEVICE
```


## Citation

If you find SCEIR is useful in your research, please cite our paper: