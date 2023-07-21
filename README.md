# Neural Video Depth Stabilizer (ICCV2023) 🚀🚀🚀

🎉🎉🎉 **Welcome to the NVDS GitHub repository!** 🎉🎉🎉  

**The repository is official PyTorch implementation of ICCV2023 paper "Neural Video Depth Stabilizer" (NVDS).**

Authors: [Yiran Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=p_RnaI8AAAAJ)<sup>1</sup>,
[Min Shi](https://www.semanticscholar.org/author/Min-Shi/1516268415)<sup>1</sup>,
[Jiaqi Li](https://scholar.google.com/citations?hl=zh-CN&user=i-2ghuYAAAAJ)<sup>1</sup>,
[Zihao Huang](https://orcid.org/0000-0002-8804-191X)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>3*</sup>,
[Guosheng Lin](https://guosheng.github.io/)<sup>3</sup>


Institutes: <sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research, <sup>3</sup>Nanyang Technological University

### [Project Page](https://raymondwang987.github.io/NVDS/) | [Arxiv](https://arxiv.org/abs/2307.08695) | [Video](https://youtu.be/SNV9F-60xrE) | [视频](https://www.bilibili.com/video/BV1KW4y1d7F8/) | [Supp](https://github.com/RaymondWang987/NVDS/blob/main/PDF/SUPPV1.pdf) | VDW Dataset (Coming Soon)

## 😎 Highlights
**NVDS is the first plug-and-play stabilizer** that can remove flickers from any single-image depth model without extra effort. Besides, we also introduce a large-scale dataset, **Video Depth
in the Wild (VDW)**, which consists of 14,203 videos with over two million frames, making it the largest natural-scene video depth dataset. Don't forget to star this repo if you find it interesting! 

Our VDW dataset is quite large (2.23 million frames, over 8TB on hard drive). Heavy works are needed for open-source. The VDW dataset can only be used for academic and research purposes. Once we are ready, we will release NVDS model and VDW dataset for the community. Stay tuned!

## ⚡ Updates and Todo List
+ [2023.07.16] Our work is accepted by ICCV2023.
+ [2023.07.18] The [Arxiv](https://arxiv.org/abs/2307.08695) version of our NVDS paper is released.
+ [2023.07.18] Our [Project Page](https://raymondwang987.github.io/NVDS/) is built and released.
+ [TODO] Releasing NVDS demo inference code in about two weeks.
+ [TODO] More evaluation code and checkpoints will be updated in 4-6 weeks.
+ [TODO] Training code might be released along with VDW dataset, as an example of using it to train models.
+ [TODO] We will construct VDW official website and gradually release our data. Stay tuned!

##  🌼 Abstract
Video depth estimation aims to infer temporally consistent depth. Some methods achieve temporal consistency by finetuning a single-image depth model during test time using geometry and re-projection constraints, which is inefficient and not robust. An alternative approach is to learn how to enforce temporal consistency from data, but this requires well-designed models and sufficient video depth data. To address these challenges, we propose a plug-and-play framework called Neural Video Depth Stabilizer (NVDS) that stabilizes inconsistent depth estimations and can be applied to different single-image depth models without extra effort. We also introduce a large-scale dataset, Video Depth in the Wild (VDW), which consists of 14,203 videos with over two million frames, making it the largest natural-scene video depth dataset to our knowledge. We evaluate our method on the VDW dataset as well as two public benchmarks and demonstrate significant improvements in consistency, accuracy, and efficiency compared to previous approaches. Our work serves as a solid baseline and provides a data foundation for learning-based video depth models. We will release our dataset and code for future research.
<p align="center">
<img src="PDF/fig1-pipeline.PNG" width="100%">
</p>

## 🔨 Installation
+ Basic environment.
  
  Our code is based on `python=3.8.13` and `pytorch==1.9.0`. Refer to the `requirements.txt` for installation. 
  ```
  conda create -n NVDS python=3.8.13
  conda activate NVDS
  conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
  pip install numpy imageio opencv-python scipy tensorboard timm scikit-image tqdm glob h5py
  ```
+ Installation of GMflow.
  
  We utilize state-of-the-art optical flow model [GMFlow](https://arxiv.org/abs/2111.13680) in the temporal loss and the OPW metric. The temporal loss is used to enhance consistency while training. The OPW metric is evaluated in our demo (inference) code to showcase quantitative improvements. <br>Please refer to the [GMFlow Official Repo](https://github.com/haofeixu/gmflow) for the installation.

+ Installation of mmcv and mmseg.

  Cross attention in our stabilization network contains functions based on `mmcv-full==1.3.0` and `mmseg==0.11.0`.  <br>Please refer to [MMSegmentation-v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) for the installation.

## 🔥 Demo & Inference
+ Preparing Demo Videos.
  
  We put 8 demo input videos in `demo_videos` files. `bandage_1` and `market_6` are examples of [Sintel](http://sintel.is.tue.mpg.de/) dataset. `motocross-jump` is from [DAVIS](https://davischallenge.org/) dataset. Others are a few examples of our VDW test dataset. You can also prepare your own testing sequences like us.
