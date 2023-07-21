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
+ [2023.07.21] We present the NVDS checkpoint and demo (inference) code.
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

  Cross attention in our stabilization network contains functions based on `mmcv-full==1.3.0` and `mmseg==0.11.0`.  <br>Please refer to [MMSegmentation-v0.11.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0) for the installation.

## 🔥 Demo & Inference
+ Preparing Demo Videos.
  
  We put 8 demo input videos in `demo_videos` folder, in which `bandage_1` and `market_6` are examples of [MPI Sintel dataset](http://sintel.is.tue.mpg.de/). `motocross-jump` is from [DAVIS dataset](https://davischallenge.org/). Others are a few examples of our VDW test dataset. You can also prepare your own testing sequences like us.

+ Downloading checkpoints of depth predictors.

  In our demo, we adopt [MiDaS](https://github.com/isl-org/MiDaS) and [DPT](https://github.com/isl-org/DPT) as different depth predictors. We use [midas_v21-f6b98070.pt](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt) and [dpt_large-midas-2f21e586.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt). Download those checkpoints and put them in `dpt/checkpoints/` folder. You may need to modify the MiDaS checkpoint name (midas_v21_384.pt) or our code (midas_v21-f6b98070.pt) since its name is adjusted by the [MiDaS repo](https://github.com/isl-org/MiDaS).

+ Preparing checkpoint of NVDS Stabilizer.
  
  [Download](https://github.com/RaymondWang987/NVDS/releases/tag/NVDS_checkpoints) and put the `NVDS_Stabilizer.pth` in `NVDS_checkpoints/` folder.
  
+ Running NVDS Inference Demo.

  `infer_NVDS_dpt_bi.py` and `infer_NVDS_midas_bi.py` use DPT and Midas as depth predictors. Those scripts contain: (1) NVDS Bidirectional Inference; (2) OPW Metric Evaluations with GMFlow. The only difference between those two scripts is the depth predictor. For running the code, taking DPT as an example, the basic command is:
  ```
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_dpt_bi.py --base_dir /XXX/XXX --vnum XXX --infer_w XXX --infer_h XXX
  ```
  `--base_dir` represents the folder to save results. `--vnum` refer to the video numbers or names. `--infer_w` and `--infer_h` are the width and height for inference. We use `--infer_h 384` by default. The `--infer_w` is set to maintain the aspect ratio of original videos.

  Specifically, for the videos of VDW test dataset (`000423` as an example):
  ```
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_dpt_bi.py --base_dir ./demo_outputs/dpt_init/000423/ --vnum 000423 --infer_w 896 --infer_h 384
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_midas_bi.py --base_dir ./demo_outputs/midas_init/000423/ --vnum 000423 --infer_w 896 --infer_h 384
  ```
  For the videos of Sintel dataset (`market_6` as an example):
  ```
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_dpt_bi.py --base_dir ./demo_outputs/dpt_init/market_6/ --vnum market_6 --infer_w 896 --infer_h 384
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_midas_bi.py --base_dir ./demo_outputs/midas_init/market_6/ --vnum market_6 --infer_w 896 --infer_h 384
  ``` 
  For the videos of DAVIS dataset (`motocross-jump` as an example):
  ```
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_dpt_bi.py --base_dir ./demo_outputs/dpt_init/motocross-jump/ --vnum motocross-jump --infer_w 672 --infer_h 384
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_midas_bi.py --base_dir ./demo_outputs/midas_init/motocross-jump/ --vnum motocross-jump --infer_w 672 --infer_h 384
  ```
  Under the resolution of $896\times384$, the inference of DPT-Large and our stabilizer takes about 20G and 5G GPU memory (RTX-A6000). If the memory occupancy is too large for your server, you can (1) run DPT/Midas initial depth results and our NVDS separately; (2) reduce the inference resolution ($e.g.$, $384\times384$); (3) if not needed, remove the OPW evaluations, in which the inference of GMFlow also brings some computational costs. (4) if not needed, remove the bidirectional (backward and mixing) inference. The forward inference process can also produce satisfactary results, while bidirectional inference can further improve consistency.

  After running the inference code, the result folder `--base_dir` will be organized as follows:
  ```
  demo_outputs/dpt_init/000423/
  └─── result.txt
      ├── initial/
        └── color/
        └── gray/
      ├── 1/
        └── color/
        └── gray/
      ├── 2/
        └── color/
        └── gray/
      ├── mix/
        └── color/
        └── gray/
    ```
  `result.txt` contains the OPW evaluations of initial depth (depth predictor, `initial/`), NVDS forward predictions (`1/`), backward predictions (`2/`), and final bidirectional results (`mix/`). `color` contains depth visualizations and `gray` contains depth results in uint16 format (0-65535).

  After getting the results, video comparisons can be generated and saved in `demo_outputs_videos/` by `pic2v.py`.
  ```
  python pic2v.py --vnum 000423 --infer_w 896 --infer_h 384
  python pic2v.py --vnum market_6 --infer_w 896 --infer_h 384
  python pic2v.py --vnum motocross-jump --infer_w 672 --infer_h 384
  ```
  We showcase the 8 video comparisons in the folder. The first row is RGB video, the second row is initial depth (DPT and MiDaS), and the third row is NVDS results with DPT and MiDaS as depth predictors. To ensure the correctness of your running results, you can compare the results you obtained with `demo_outputs_videos` and `demo_outputs`(png results). We also showcase in 8 png results by [LINK](https://drive.google.com/file/d/1MG13LpbRxnxGrofo1TI91ZNln9HVJmfq/view?usp=sharing). Besides, you are also encouraged to modify our code to stabilize your own depth predictors and discuss the results with us. We hope our work can serve as a solid baseline for future works in video depth estimation and other relevant tasks.   
  


