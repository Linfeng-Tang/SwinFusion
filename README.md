<h1 align="center"><a href="https://ieeexplore.ieee.org/document/9812535">SwinFusion: Cross-domain Long-range Learning for General Image Fusion via Swin Transformer</a></h1>

<p align="center"><a href="https://sites.google.com/site/jiayima2013">Jiayi Ma</a>&emsp; <a href="https://github.com/Linfeng-Tang">Linfeng Tang</a>&emsp; Fan Fan&emsp; Jun Huang&emsp; Xiaoguang Mei&emsp; Yong Ma</p>
<p align="center"><strong>Wuhan University &middot; Northwestern Polytechnical University</strong></p>
<p align="center"><strong>IEEE/CAA Journal of Automatica Sinica</strong> &middot; 2022</p>
<p align="center"><a href="https://esi.help.clarivate.com/Content/overview.htm"><img src="https://img.shields.io/badge/%F0%9F%94%A5_ESI_Hot-Top_0.1%25-E85D3F?style=flat-square" alt="ESI Hot Paper (top 0.1%)"></a> <a href="https://esi.help.clarivate.com/Content/overview.htm"><img src="https://img.shields.io/badge/%F0%9F%8F%86_ESI_Highly_Cited-Top_1%25-D4A017?style=flat-square" alt="ESI Highly Cited Paper (top 1%)"></a> <a href="https://www.ieee-jas.net/news/news_en/5e720d40-3647-459c-acd8-df750fa9f74f_en.htm"><img src="https://img.shields.io/badge/%F0%9F%8F%85_Hsue--shen_Tsien_Paper_Award-2023-7B61A8?style=flat-square" alt="Hsue-shen Tsien Paper Award 2023"></a><br><sub><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=PyRqpAsAAAAJ&citation_for_view=PyRqpAsAAAAJ:u-x6o8ySG0sC">Google Scholar &middot; <strong>1,582 citations</strong></a> &middot; updated July 18, 2026</sub></p>

## ✨ News  
- **[2026-06-02]** Our paper **[DSPFusion: Image Fusion via Degradation and Semantic Dual-Prior Guidance](https://doi.org/10.1109/TIP.2026.3700938)** has been officially accepted by **IEEE Transactions on Image Processing (IEEE TIP)**! [[Paper](https://doi.org/10.1109/TIP.2026.3700938)] [[arXiv](https://arxiv.org/abs/2503.23355)] [[Code](https://github.com/Linfeng-Tang/DSPFusion)]
- **[2026-02-21]** Our paper **[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_VideoFusion_A_Spatio-Temporal_Collaborative_Network_for_Multi-modal_Video_Fusion_CVPR_2026_paper.html)** has been accepted by **The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)**! [[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_VideoFusion_A_Spatio-Temporal_Collaborative_Network_for_Multi-modal_Video_Fusion_CVPR_2026_paper.html)] [[arXiv](https://arxiv.org/abs/2503.23359)] [[Code](https://github.com/Linfeng-Tang/VideoFusion)]
- **[2025-09-18]** Our paper *[ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](https://arxiv.org/pdf/2503.23356?)* has been officially accepted by **Advances in Neural Information Processing Systems (NeurIPS 2025)**! [[Paper](https://arxiv.org/pdf/2503.23356?)] [[Code](https://github.com/Linfeng-Tang/ControlFusion)]  

- **[2025-09-10]** Our paper *[Mask-DiFuser: A Masked Diffusion Model for Unified Unsupervised Image Fusion](https://ieeexplore.ieee.org/document/11162636)* has been officially accepted by **IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI)**! [[Paper](https://ieeexplore.ieee.org/document/11162636)] [[Code](https://github.com/Linfeng-Tang/Mask-DiFuser)]  

- **[2025-03-15]** Our paper *[C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning](https://github.com/Linfeng-Tang/C2RF)* has been officially accepted by the **International Journal of Computer Vision (IJCV)**! [[Paper](https://link.springer.com/article/10.1007/s11263-025-02427-1)] [[Code](https://github.com/Linfeng-Tang/C2RF)]  

- **[2025-02-11]** We released a large-scale dataset for infrared and visible video fusion: *[M3SVD: Multi-Modal Multi-Scene Video Dataset](https://github.com/Linfeng-Tang/M3SVD)*.  

## Image Fusion Example
![Schematic illustration of multi-modal image fusion and digital photography image fusion. ](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/Schematic_illustration.jpg)
Schematic illustration of multi-modal image fusion and digital photography image fusion. First row: source image pairs, second row: fused results of U2Fusion and our SwinFusion.

## Framework
![The framework of the proposed SwinFusion for multi-modal image fusion and digital photography image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/SwinFusion1.jpg)
The framework of the proposed SwinFusion for multi-modal image fusion and digital photography image fusion.

## Visible and Infrared Image Fusion (VIF)
### To Train
Download the training dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/trainsets/MSRS/**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_swinfusion.py --opt options/swinir/train_swinfusion_vif.json  --dist True

### To Test
Download the test dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/testsets/MSRS/**. 

    python test_swinfusion.py --model_path=./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/ --iter_number=10000 --dataset=MSRS --A_dir=IR  --B_dir=VI_Y
 
 ### Visual Comparison
![Qualitative comparison of SwinFusion with five state-of-the-art methods on visible and infrared image fusion](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/VIF.jpg)
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
![Qualitative comparison of SwinFusion with five state-of-the-art methods on visible and near-infrared image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/NIR.jpg)
Qualitative comparison of SwinFusion with five state-of-the-art methods on visible and near-infrared image fusion. From left to right: near-infrared
image, visible image, and the results of ANVF, DenseFuse, IFCNN, SDNet, U2Fusion, and our SwinFusion.

## Medical Image Fusion (Med)
### To Train
Download the training dataset from [**Harvard medical dataset**](http://www.med.harvard.edu/AANLIB/home.html), and put it in **./Dataset/trainsets/PET-MRI/** or **./Dataset/trainsets/CT-MRI/**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_swinfusion.py --opt options/swinir/train_swinfusion_med.json  --dist True
    
### To Test
Download the training dataset from [**Harvard medical dataset**](http://matthewalunbrown.com/nirscene/nirscene.html), and put it in **./Dataset/testsets/PET-MRI/** or **./Dataset/testsets/CT-MRI/**. 

    python test_swinfusion.py --model_path=./Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/  --iter_number=10000 --dataset=NirScene --A_dir=MRI --B_dir=PET_Y
**or** 

    python test_swinfusion.py --model_path=./Model/Medical_Fusion-CT-MRI/Medical_Fusion/models/ --iter_number=10000 --dataset=CT-MRI--A_dir=MRI --B_dir=CT

### Visual Comparison
![Qualitative comparison of SwinFusion with five state-of-the-art methods on PET and MRI image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/PET-MRI.jpg)
Qualitative comparison of SwinFusion with five state-of-the-art methods on PET and MRI image fusion. From left to right: MRI image, PET image,
and the results of CSMCA, DDcGAN, IFCNN, SDNet, U2Fusion, and our SwinFusion.

![Qualitative comparison of SwinFusion with five state-of-the-art methods on CT and MRI image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/CT-MRI.jpg)
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
![Qualitative results of multi-exposure image fusion. ](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/MEF.jpg)
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
![Qualitative results of multi-focus image fusion.](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/MFF.jpg)
Qualitative results of multi-focus image fusion. From left to right: near/far-focus image, the fused results and difference maps of SFMD, DRPL,
MFF-GAN, IFCNN, SDNet, U2Fusion, and our SwinFusion. The difference maps represent the difference between the near-focus image and fused results.


## Recommended Environment

 - [x] torch 1.11.0
 - [x] torchvision 0.12.0
 - [x] tensorboard  2.7.0
 - [x] numpy 1.21.2

## Citation
```
@article{Tang2024Mask-DiFuser,
  author={Tang, Linfeng and Li, Chunyu and Ma, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Mask-DiFuser: A Masked Diffusion Model for Unified Unsupervised Image Fusion}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
 }
```

```
@article{Tang2024C2RF,
	title={C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning}, 
	author={Tang, Linfeng and Yan, Qinglong and Xiang, Xinyu and Fang, Leyuan and Ma, Jiayi},
	journal={International Journal of Computer Vision}, 
	pages={5262--5280},
	volume={133},
	year={2025},
}
```

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
