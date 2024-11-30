# 3D-mADUNet: Multi-resolution Guided 3D GANs for Medical Image Translation
Juhyung Ha\*, Jong Sung Park, David Crandall, Eleftherios Garyfallidis, Xuhong Zhang


## [Project page](https://juhha.github.io/3D-mADUNet-page/) | [Paper (WACV25)]() | [ArXiv]()
This is GitHub repo for "Multi-resolution Guided 3D GANs for Medical Image Translation". Codes includes model training, evaluation, visualization, and calculating/ aggregating the model performance.

For the reproducing the model performance, we provide detailed results for IQA and Dice analysis (detailed performance saved in `results` folder and example aggregation codes are in `example` folder - `evaluate_lpips.ipynb`, `show_iqa_result.ipynb`, `dice_brats.ipynb`, `dice_synth.ipynb`)

For the pre-trained weights and sample data, you can download here on [Google Drive](https://drive.google.com/drive/folders/1v8r-lIhilv-UOl4d7mSSImr03jfDsBVo?usp=sharing). To generate new sample visualization with sample data (`example/sample_inference.ipynb`), please download these first and store in `data/samples` for data sample and in `checkpoint` for pre-trained models.

For training with the same dataset, you should download the dataset first (links are in below).

For training with the custom dataset, you should match the file paths in meta json file stored in `data/meta`.

## Data Preparation
For data I/O, the list of files should be saved as json meta file located in `data/meta` folder. See `data/meta/brats21.json` as reference.

### HCP1200 & dHCP
This is public-upon-approval dataset. Go to [hcp1200](https://www.humanconnectome.org) and [dHCP](https://www.developingconnectome.org/project) for details

### BraTS2021
This is public dataset that can be downloaded [here](https://www.med.upenn.edu/cbica/brats2021/#Data2)

### SynthRAD2023
This is a challenge dataset from [link](https://synthrad2023.grand-challenge.org).

## Train
```
python train.py --config_file {config_file_name} --fold {fold_id} --persistent --device {device_name} --progress pbar
```
* `--persistent`: optional for faster I/O data loader. But this takes space in hard disk
* `--progress pbar`: optional when you want to see the progress bar in print
* `--config_file`: configuration file (without yaml extension) in `options` folder
* `--fold`: fold ID. In our study, we split data into 75/25, meaning it can be one of 0,1,2,3
* `--device`: Device name. If cuda is available, use "cuda"

Example codes are below:
```
python train.py --config_file brats_t2toflair --fold 0 --persistent --device "cuda" --progress pbar
```

After the training starts, progress log will be saved in `checkpoint` folder. See `train_samples` subfolder to see the progress in sample visualization.

### Configuration
In configuration file in yaml, there are options largely for (1) data, (2) training parameters, (3) network parameters, (4) loss, (5) hyperparameters (optimizer, scheduler, mixed-precision). For (2), (3), (4), and (5), please refer `options/brats_t2toflair.yaml` for our experiment settings.
* Data Parameter (`data_opt`): This is configuration for data. Data processing options should be different depending on imaging modalities. In our study, we applied min-max scaling for MRIs, and min-max scaling after intensity clipping for CBCT/CT.
* Training Parameter (`train_opt`, `trainer_opt`):
    * `train_opt`: this includes `n` for cross-validation, `patch_size` for size of 3D patch for data processing ([96,96,96] is used in our study), `num_patch` and `batch_size` are used for batch size of the training (essentially how many number of patches are included in a batch), and others are self-explanatory.
    * `trainer_opt`: use "srgan" for GAN training and "sr" for non-GAN training. In our study, we used "srgan".
* Network Parameter (`network_opt`)
    * `net_g`: network parameters for generator module.
    * `net_d`: network parameters for discriminator module.
* Loss (`loss_opt`)
    * `generator`: loss for generator module. This includes pixel_loss, perception_loss, and adv_loss.
    * `discriminator` : loss for discriminator module
* Other Hyperparameters (`optim_opt`, `scheduler_opt`, `scaler_opt`, `autocast`)
    * Please refer `brats_t2toflair` for our experiment settings for detail.

## Inference
```
python inference.py --config_file {config_file_name} --fold {fold_id} --num_workers {# of thread} --persistent --device {device_name}
```
Again, note that `--persistent` is optional. If you have done `--persistent` with training already, you can have this for faster I/O.

Example codes are below:
```
python inference.py --config_file brats_t2toflair --fold 0 --num_workers 8 --persistent --device "cuda:0"
```

### Inference using pre-trained models
Refer `example/sample_inference.ipynb` for inferencing using pre-trained model. This notebook have all 4 experiments with sample data.

## Evaluate
Before evaluation, there should be synthesize image paired with ground-truth.
### Image Quality Assessment (IQA)
For evaluation codes for IQA, refer `evaluate_iqa.py`, `example/show_iqa_result.ipynb`, and `evaluate_lpips.ipynb`.
#### SSIM, PSNR, NMSE
```
python evaluate_iqa.py --gt_dir {gt_image_directory} --pred_dir {synthetic_image_directory} --result_path {json_result_path} --target_modality {target_modality} --source_modality {source_modality}
```
Example codes are below:
```
python evaluate_iqa.py --gt_dir ../data --pred_dir ./checkpoint/brats_t2toflair/fold_0/test_output --result_path ./results/example.json --target_modality flair --source_modality t2
```

All of SSIM, PSNR, and NMSE are calculated in 3D (see `evaluate_iqa.py`).

#### LPIPS
The process of calculating LPIPS is more complicated than other IQAs. For detailed steps, please refer `example/evaluate_lpips.ipynb`. The steps involve as following. 
    1. Generate 2D slides from 3D images (for both synthetic and ground-truth) - each 2D slides in 3 directions (sagittal, coronal, axial) are generated and saved
    2. Calcualte LPIPS for paired 2D slices - this results are calculated and saved in `results/lpips` folder.
    3. Aggregate the scores by average and standard deviation

## Comments for training
It's important to keep monitoring "train_samples" in checkpoint during training. If only blank images (all-black) images are generated in initial stage (in epoch 1), drop training, delete checkpoint, and re-run training. In patch-wise training 3D images using GAN, there are 2 random factors that can negatively impact: (1) random patch selection and (2) instability of GAN itself. For example, (1) can generate some bad samples (e.g. background-only image), then GAN training can be very instable after.

To avoid the blank image, try using `relativistic: False` for adv_loss and disc_loss in configuration file. We found relativistic loss helps generating better images, but not stable (`relativistic: False` will give slightly worse results in terms of quantitative analysis, but it is more stable).