
# SwinFusion
This is official Pytorch implementation of "SwinFusion: Cross-domain Long-range Learning for General Image Fusion via Swin Transformer"
## Image Fusion Example
![Schematic illustration of multi-modal image fusion and digital photography image fusion. ](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/Schematic_illustration.png)
Schematic illustration of multi-modal image fusion and digital photography image fusion. First row: source image pairs, second row: fused results of U2Fusion and our SwinFusion.

## Framework
![The framework of the proposed SwinFusion for multi-modal image fusion and digital photography image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/SwinFusion1.png)
The framework of the proposed SwinFusion for multi-modal image fusion and digital photography image fusion.

## Visible and Infrared Image Fusion (VIF)
### To Train
Download the training dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/trainsets/MSRS/**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_swinfusion.py --opt options/swinir/train_swinfusion_vif.json  --dist True

### To Test
Download the test dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/testsets/MSRS/**. 

    python test_swinfusion.py --model_path=./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/ --iter_number=10000 --dataset=MSRS --A_dir=IR  --B_dir=VI_Y
 
 ### Visual Comparison
![Qualitative comparison of SwinFusion with five state-of-the-art methods on visible and infrared image fusion](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/VIF.png)
Qualitative comparison of SwinFusion with five state-of-the-art methods on visible and infrared image fusion. From left to right: infrared image, visible
image, and the results of GTF, DenseFuse, IFCNN SDNet, U2Fusion, and our SwinFusion.

## Visible and Nir-infrared Image Fusion (VIS-NIR)
### To Train
Download the training dataset from [**VIS-NIR Scene dataset**](http://matthewalunbrown.com/nirscene/nirscene.html), and put it in **./Dataset/trainsets/Nirscene/**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_swinfusion.py --opt options/swinir/train_swinfusion_nir.json  --dist True

### To Test
Download the test dataset from [**VIS-NIR Scene dataset**](http://matthewalunbrown.com/nirscene/nirscene.html), and put it in **./Dataset/testsets/Nirscene/**. 

    python test_swinfusion.py --model_path=./Model/RGB_NIR_Fusion/RGB_NIR_Fusion/models/ --iter_number=10000 --dataset=NirScene --A_dir=NIR  --B_dir=VI_Y

### Visual Comparison
![Qualitative comparison of SwinFusion with five state-of-the-art methods on visible and near-infrared image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/NIR.png)
Qualitative comparison of SwinFusion with five state-of-the-art methods on visible and near-infrared image fusion. From left to right: near-infrared
image, visible image, and the results of ANVF, DenseFuse, IFCNN, SDNet, U2Fusion, and our SwinFusion.

## Medical Image Fusion (Med)
### To Train
Download the training dataset from [**Harvard medical dataset**](http://matthewalunbrown.com/nirscene/nirscene.html), and put it in **./Dataset/trainsets/PET-MRI/** or **./Dataset/trainsets/CT-MRI/**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_swinfusion.py --opt options/swinir/train_swinfusion_med.json  --dist True
    
### To Test
Download the training dataset from [**Harvard medical dataset**](http://matthewalunbrown.com/nirscene/nirscene.html), and put it in **./Dataset/testsets/PET-MRI/** or **./Dataset/testsets/CT-MRI/**. 

    python test_swinfusion.py --model_path=./Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/  --iter_number=10000 --dataset=NirScene --A_dir=MRI --B_dir=PET_Y
**or** 

    python test_swinfusion.py --model_path=./Model/Medical_Fusion-CT-MRI/Medical_Fusion/models/ --iter_number=10000 --dataset=CT-MRI--A_dir=MRI --B_dir=CT

### Visual Comparison
![Qualitative comparison of SwinFusion with five state-of-the-art methods on PET and MRI image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/PET-MRI.png)
Qualitative comparison of SwinFusion with five state-of-the-art methods on PET and MRI image fusion. From left to right: MRI image, PET image,
and the results of CSMCA, DDcGAN, IFCNN, SDNet, U2Fusion, and our SwinFusion.

![Qualitative comparison of SwinFusion with five state-of-the-art methods on CT and MRI image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/CT-MRI.png)
Qualitative comparison of SwinFusion with five state-of-the-art methods on CT and MRI image fusion. From left to right: MRI image, CT image, and
the results of CSMCA, DDcGAN, IFCNN, SDNet, U2Fusion, and our SwinFusion.

## Multi-Exposure Image Fusion (MEF)
### To Train
Download the training dataset from [**MEF dataset**](https://github.com/csjcai/SICE), and put it in **./Dataset/trainsets/MEF**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_swinfusion.py --opt options/swinir/train_swinfusion_mef.json  --dist True

### To Test
Download the training dataset from [**MEF Benchmark dataset**](https://github.com/xingchenzhang/MEFB), and put it in **./Dataset/testsets/MEF_Benchmark**. 

    python test_swinfusion.py --model_path=./Model/Multi_Exposure_Fusion/Multi_Exposure_Fusion/models/ --iter_number=10000 --dataset=MEF_Benchmark --A_dir=under_Y --B_dir=over_Y
    
### Visual Comparison
![Qualitative results of multi-exposure image fusion. ](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/MEF.png)
Qualitative results of multi-exposure image fusion. From left to right: under-exposed image, over-exposed image, and the results of SPD-MEF,
MEF-GAN, IFCNN SDNet, U2Fusion, and our SwinFusion.


## Multi-Focus Image Fusion (MFF)
### To Train
Download the training dataset from [**MFI-WHU dataset**](https://github.com/HaoZhang1018/MFI-WHU), and put it in **./Dataset/trainsets/MEF**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_swinfusion.py --opt options/swinir/train_swinfusion_mff.json  --dist True
### To Test
Download the training dataset from [**Lytro dataset**](https://github.com/HaoZhang1018/MFI-WHU), and put it in **./Dataset/trainsets/Lytro**. 

    python test_swinfusion.py --model_path=./Model/Multi_Focus_Fusion/Multi_Focus_Fusion/models/ --iter_number=10000 --dataset=Lytro --A_dir=A_Y --B_dir=B_Y
    
### Visual Comparison
![Qualitative results of multi-focus image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/Figure/MFF.png)
Qualitative results of multi-focus image fusion. From left to right: near/far-focus image, the fused results and difference maps of SFMD, DRPL,
MFF-GAN, IFCNN, SDNet, U2Fusion, and our SwinFusion. The difference maps represent the difference between the near-focus image and fused results.


## Recommended Environment

 - [x] torch 1.11.0
 - [x] torchvision 0.12.0
 - [x] tensorboard  2.7.0
 - [x] numpy 1.21.2

## Citation
```
@article{Ma2022SwinFusion,  
author={Ma, Jiayi and Tang, Linfeng and Fan, Fan and Huang, Jun and Mei, Xiaoguang and Ma, Yong},  
journal={IEEE/CAA Journal of Automatica Sinica},   
title={SwinFusion: Cross-domain Long-range Learning for General Image Fusion via Swin Transformer},   
year={2022},  
volume={9},  
number={7},  
pages={1200-1217}
}
```
## Acknowledgement
The codes are heavily based on [SwinIR](https://github.com/JingyunLiang/SwinIR). Please also follow their licenses. Thanks for their awesome works.
