# NVDS (ICCV 2023) & NVDS+ (TPAMI 2024) 🚀🚀🚀

🎉🎉🎉 **Welcome to the NVDS GitHub repository!** 🎉🎉🎉  

**The repository is official PyTorch implementation of ICCV 2023 paper "Neural Video Depth Stabilizer" (NVDS)**

Authors: [Yiran Wang](https://raymondwang987.github.io)<sup>1</sup>,
[Min Shi](https://www.semanticscholar.org/author/Min-Shi/1516268415)<sup>1</sup>,
[Jiaqi Li](https://scholar.google.com/citations?hl=zh-CN&user=i-2ghuYAAAAJ)<sup>1</sup>,
[Zihao Huang](https://orcid.org/0000-0002-8804-191X)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>,
[Ke Xian](https://kexianhust.github.io)<sup>3*</sup>, 
[Guosheng Lin](https://guosheng.github.io/)<sup>3</sup>


Institutes: <sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research, <sup>3</sup>Nanyang Technological University

### [Project Page](https://raymondwang987.github.io/NVDS/) | [Arxiv](https://arxiv.org/abs/2307.08695) | [Video](https://youtu.be/SNV9F-60xrE) | [视频](https://www.bilibili.com/video/BV1KW4y1d7F8/) | [Poster](https://github.com/RaymondWang987/NVDS/blob/main/PDF/NVDS_Poster_ICCV23.pdf) | [Supp](https://github.com/RaymondWang987/NVDS/blob/main/PDF/camera_ready/NVDS_supp.pdf) | [VDW Dataset](https://raymondwang987.github.io/VDW/) | [VDW Toolkits](https://github.com/RaymondWang987/VDW_Dataset_Toolkits)

**and TPAMI 2024 paper "NVDS+: Towards Efficient and Versatile Neural Stabilizer for Video Depth Estimation" (NVDS+)**

Authors: [Yiran Wang](https://raymondwang987.github.io)<sup>1</sup>,
[Min Shi](https://www.semanticscholar.org/author/Min-Shi/1516268415)<sup>1</sup>,
[Jiaqi Li](https://scholar.google.com/citations?hl=zh-CN&user=i-2ghuYAAAAJ)<sup>1</sup>,
[Chaoyi Hong](https://scholar.google.com.hk/citations?hl=zh-CN&user=N9YzPMcAAAAJ)<sup>1</sup>,
[Zihao Huang](https://orcid.org/0000-0002-8804-191X)<sup>1</sup>,
[Juewen Peng](https://juewenpeng.github.io)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>,
[Ke Xian](https://kexianhust.github.io)<sup>3*</sup>, 
[Guosheng Lin](https://guosheng.github.io/)<sup>3</sup>


Institutes: <sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research, <sup>3</sup>Nanyang Technological University

### [Project Page](https://raymondwang987.github.io/NVDS/) | Arxiv (Coming Soon) | [Video](https://www.youtube.com/watch?v=L-yeR_aki20) | [Supp](https://github.com/RaymondWang987/NVDS/blob/main/PDF/Supp_NVDS_Plus.pdf) | [VDW Dataset](https://raymondwang987.github.io/VDW/) | [VDW Toolkits](https://github.com/RaymondWang987/VDW_Dataset_Toolkits)

## 😎 Highlights
**NVDS is the first plug-and-play stabilizer** that can remove flickers from any single-image depth model without extra effort. Besides, we also introduce a large-scale dataset, **Video Depth
in the Wild (VDW)**, which consists of 14,203 videos with over two million frames, making it the largest natural-scene video depth dataset. Don't forget to star this repo if you find it interesting! 

## 💦 License and Releasing Policy
+ VDW dataset.

  We have released the VDW dataset under strict conditions. We must ensure that the releasing won’t violate any copyright requirements. **To this end, we will not release any video frames or the derived data in public.** Instead, we provide meta data and detailed toolkits, which can be used to reproduce VDW or generate your own data. The meta data contains [IMDB](https://www.imdb.com/) numbers, starting time, end time, movie durations, resolutions, and cropping areas. All the meta data and toolkits are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), which can only be used for academic and research purposes. Please refer to our [VDW official website](https://raymondwang987.github.io/VDW/) and [VDW Toolkits](https://github.com/RaymondWang987/VDW_Dataset_Toolkits) for data usage.

+ NVDS code and model.
  
  Following [MiDaS](https://github.com/isl-org/MiDaS) and [CVD](https://github.com/facebookresearch/consistent_depth), NVDS model adopts the widely-used [MIT License](https://github.com/RaymondWang987/NVDS/blob/main/LICENSE).

## ⚡ Updates and Todo List
+ [TODO] The paper and code of **NVDS+** will be gradually released.
+ [2024.10.02] The extension **NVDS+：Towards Efficient and Versatile Neural Stabilizer for Video Depth Estimation** is accepted by TPAMI 2024!.
+ [2024.06.03] The [VDW official toolkits](https://github.com/RaymondWang987/VDW_Dataset_Toolkits) to reproduce VDW and generate your own data.
+ [2024.01.22] We release the [supplementary video](https://www.youtube.com/watch?v=L-yeR_aki20) for the journal extension from **NVDS to NVDS+**.
+ [2024.01.22] The metadata and evaluation code of the VDW test set.
+ [2023.09.17] Upload [NVDS Poster](https://github.com/RaymondWang987/NVDS/blob/main/PDF/NVDS_Poster_ICCV23.pdf) for [ICCV2023](https://iccv2023.thecvf.com/).
+ [2023.09.09] Evaluation code on VDW test set is released.
+ [2023.09.09] [VDW official website](https://raymondwang987.github.io/VDW/) goes online.
+ [2023.08.11] Release evaluation code and checkpoint of [NYUDV2-finetuned NVDS](https://github.com/RaymondWang987/NVDS/releases/tag/NVDS-finetuned-NYUDV2).
+ [2023.08.10] Update the camera ready version of NVDS [paper](https://github.com/RaymondWang987/NVDS/blob/main/PDF/camera_ready/NVDS_camera.pdf) and [supplementary](https://github.com/RaymondWang987/NVDS/blob/main/PDF/camera_ready/NVDS_supp.pdf).
+ [2023.08.05] Update license of VDW dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
+ [2023.07.21] We present the [NVDS checkpoint](https://github.com/RaymondWang987/NVDS/releases/tag/NVDS_checkpoints) and demo (inference) code.
+ [2023.07.18] Our [Project Page](https://raymondwang987.github.io/NVDS/) is built and released.
+ [2023.07.18] The [Arxiv](https://arxiv.org/abs/2307.08695) version of our NVDS paper is released.
+ [2023.07.16] Our work is accepted by ICCV2023.

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

  Cross attention in our stabilization network contains functions based on `mmcv-full==1.3.0` and `mmseg==0.11.0`. Please refer to [MMSegmentation-v0.11.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0) and their [official document](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) for detailed installation instructions step by step. **The key is to match the version of mmcv-full and mmsegmentation with the version of cuda and pytorch on your server.** For instance, I have `CUDA 11.1` and `PyTorch 1.9.0` on my server, thus `mmcv-full 1.3.x` and `mmseg 0.11.0` (as in our installation instructions) are compatible with my environment (confirmed by [mmcv-full 1.3.x](https://mmcv.readthedocs.io/zh_CN/v1.3.13/get_started/installation.html)). Different servers adopt different Cuda versions, thus I can not specify the specific installation for all people. You should check the matching version of your own server on the official documents of [mmcv-full](https://mmcv.readthedocs.io/en/latest/) and [mmseg](https://mmsegmentation.readthedocs.io/en/latest/). You can choose different versions in their documents and check the version matching relations. By reading and following the detailed mmcv-full and mmseg documents, the installation seems to be easy. You can also refer to [Issue #1](https://github.com/RaymondWang987/NVDS/issues/1) for some discussions.

  Besides, **we suggest you to install `mmcv-full==1.x.x`**, because some API or functions are removed in `mmcv-full==2.x.x` (you need to adjust our code for mmcv-full==2.x.x). 

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
  `--base_dir` represents the folder to save results. `--vnum` refer to the video numbers or names. `--infer_w` and `--infer_h` are the width and height for inference. We use `--infer_h 384` by default. The `--infer_w` is set to maintain the aspect ratio of original videos. Besides, the `--infer_w` and `--infer_h` should be set to integer multiples of `32` for alignment of resolutions in the up-sampling and down-sampling processes.

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
  Under the resolution of $896\times384$, the inference of DPT-Large and our stabilizer takes about 20G and 5G GPU memory (RTX-A6000). If the GPU memory or inference latency seems large for your applications, you can (1) run DPT/Midas initial depth results and our NVDS separately; (2) reduce the inference resolution ($e.g.$, $384\times384$); (3) if not needed, remove the OPW evaluations, in which the inference of GMFlow also brings some computational costs. (4) if not needed, remove the bidirectional (backward and mixing) inference. The forward inference process can also produce satisfactory results, while bidirectional inference can further improve consistency.

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

+ Video Comparisons.
  
  After getting the results, video comparisons can be generated and saved in `demo_outputs_videos/`:
  ```
  python pic2v.py --vnum 000423 --infer_w 896 --infer_h 384
  python pic2v.py --vnum market_6 --infer_w 896 --infer_h 384
  python pic2v.py --vnum motocross-jump --infer_w 672 --infer_h 384
  ```
  We show 8 video comparisons in `demo_outputs_videos/`. The first row is RGB video, the second row is initial depth (DPT and MiDaS), and the third row is NVDS results with DPT and MiDaS as depth predictors. To ensure the correctness of your running results, you can compare the results you obtained with `demo_outputs_videos` and `demo_outputs`(png results). We show png results of the 8 videos by [LINK](https://drive.google.com/file/d/1MG13LpbRxnxGrofo1TI91ZNln9HVJmfq/view?usp=sharing). Besides, you are also encouraged to modify our code to stabilize your own depth predictors and discuss the results with us. We hope our work can serve as a solid baseline for future works in video depth estimation and other relevant tasks.

## 🍔 Evaluations on NYUDV2
+ Preparing 654 testing sequences.

  Download the 654 testing sequences from [LINK](https://www.dropbox.com/sh/noirpejsu6c91bp/AABXw4c4nqhjQNgPpA6GZg6Sa?dl=0). Put the sequences in the `./test_nyu_data` folder. The `./test_nyu_data` folder should only contain the 654 folders of all testing sequences. The folder of each sequence is organized by:
  ```
  test_nyu_data/1/
      ├── rgb/
        └── 000000.png 000001.png 000002.png 000003.png
      ├── gt/
        └── 000003.png
    ```
  We follow the commonly-applied Eigen split with 654 images for testing. In our case, we locate each image `(000003.png)` in its video and use its previous three frames `(000000.png, 000001.png, and 000002.png)` as reference frames.

+ Preparing NVDS checkpoint finetuned on NYUDV2.

  [Download](https://github.com/RaymondWang987/NVDS/releases/tag/NVDS-finetuned-NYUDV2) and put the `NVDS_Stabilizer_NYUDV2_Finetuned.pth
` in `NVDS_checkpoints/` folder.

+ Evaluations with Midas and DPT as different depth predictors.

  Run `test_NYU_depth_metrics.py` with specified depth predictors (`--initial_type dpt` or `midas`).
  ```
  CUDA_VISIBLE_DEVICES=0 python test_NYU_depth_metrics.py --initial_type dpt
  CUDA_VISIBLE_DEVICES=1 python test_NYU_depth_metrics.py --initial_type midas
  ```
  The `test_NYU_depth_metrics.py` contains three parts: (1) Inference of depth predictors, producing initial results of Midas or DPT; (2) Inference of NVDS based on the initial results; (3) Metric evaluations of depth predictor and NVDS. All inference processes are conducted by the resolution of $384\times384$ as Midas and DPT. For simplicity, we only adopt NVDS forward prediction in this code. By running the code, you can reproduce similar results as our paper:

   
  | Methods | $\delta_1$ | $Rel$ | Methods | $\delta_1$ | $Rel$ |
  | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
  | [Midas](https://github.com/isl-org/MiDaS) | 0.910 | 0.095 | [DPT](https://github.com/isl-org/DPT) | $0.928$ | $0.084$ |
  | **NVDS (Midas)** | **0.941** | **0.076** | **NVDS (DPT)** | **0.950** | **0.072** |

 
  After running the evaluation code, the `test_nyu_data` will be organized by:
   ```
  test_nyu_data/1/
      ├── rgb/
        └── 000000.png 000001.png 000002.png 000003.png
      ├── gt/
        └── 000003.png
      ├── initial_midas/
        └── 000000.png 000001.png 000002.png 000003.png
      ├── initial_dpt/
        └── 000000.png 000001.png 000002.png 000003.png
      ├── NVDS_midas/
        └── 000003.png
      ├── NVDS_dpt/
        └── 000003.png
    ```
   We evaluate depth metrics of all methods only using the 654 images in Eigen split, i.e., `000003.png` of each sequence. `000000.png, 000001.png, and 000002.png` are produced by depth predictors as the input of the stabilization network.
  
## 🎯 Evaluations on VDW Test Set
+ Applying for the VDW test set.

  Here we take `/xxx/vdw_test` as an example. The VDW test set contains 90 videos with 12,622 frames. For each video (e.g., `/xxx/vdw_test/000008/`), the test set is organized as follows. The `left` or `right` folders contain the RGB video frames of left and right views, while gt folders are for disparity annotations and mask folders for valid masks.
  ```
  /xxx/vdw_test/000008/
      ├── left/
        └── frame_000000.png frame_000001.png frame_000002.png ...
      ├── left_gt/
        └── frame_000000.png frame_000001.png frame_000002.png ...
      ├── left_mask/
        └── frame_000000.png frame_000001.png frame_000002.png ...
      ├── right/
        └── frame_000000.png frame_000001.png frame_000002.png ...
      ├── right_gt/
        └── frame_000000.png frame_000001.png frame_000002.png ...
      ├── right_mask/
        └── frame_000000.png frame_000001.png frame_000002.png ...
    ```

+ Inference and evaluations for each test video.

    For each test video, the evaluations contain two steps: **(1) inference; and (2) depth metrics evaluations**. We provide the `write_sh.py` to generate evaluation scripts (for Midas and DPT). You should modify some details in `write_sh.py` (e.g.,  gpu number, VDW test set path, directory for saving NVDS results with Midas/DPT) in order to generate the `test_VDW_NVDS_Midas.sh` and `test_VDW_NVDS_DPT.sh`. We provide the two example scripts with `/xxx/` for those directories.

    To be specific, **(1) the inference step** is the same as the previous `Demo & Inference` part with  `infer_NVDS_dpt_bi.py` and `infer_NVDS_midas_bi.py`. In this step, the temporal metric `OPW` is automatically evaluated and saved in the `result.txt`. **(2) Depth metrics evaluations** utilize the `vdw_test_metric.py` to calculate $\delta_1$ and $Rel$ for each video. Taking `./vdw_test/000008/` as an example, `--gt_dir` specifies the path for `vdw_test`, `--result_dir` specifies your directory for saving results, and `--vnum` represents the video number. 
   ```
   python vdw_test_metric.py --gt_dir /xxx/vdw_test/ --result_dir /xxx/NVDS_VDW_Test/Midas/ --vnum 000008
   python vdw_test_metric.py --gt_dir /xxx/vdw_test/ --result_dir /xxx/NVDS_VDW_Test/DPT/ --vnum 000008
   ```

   After generating `test_VDW_NVDS_Midas.sh` and `test_VDW_NVDS_DPT.sh`, you can run inference and evaluations for all the videos by:
   ```
  bash test_VDW_NVDS_Midas.sh
  bash test_VDW_NVDS_DPT.sh
   ```
+ Average metrics calculations for all 90 videos.
  
   When the scripts are finished for all videos, `NVDS_VDW_Test` folder will contain the results of 90 test videos with Midas/DPT as depth predictors (`/xxx/NVDS_VDW_Test/Midas/` and `/xxx/NVDS_VDW_Test/DPT/`). For each video, there will be an `accuracy.txt` to store the depth metrics. The last step is to calculate the average temporal and depth metrics for all the 90 videos. You can simply run the `cal_mean_vdw_metric.py` for the final results.
   ```
   python cal_mean_vdw_metric --test_dir /xxx/NVDS_VDW_Test/Midas/
   python cal_mean_vdw_metric --test_dir /xxx/NVDS_VDW_Test/DPT/
   ```

   Finally, you can get the same results as our paper. This also serves as an example to conduct evaluations on the VDW test set.

  | Methods | $\delta_1$ | $Rel$ | $OPW$  | Methods | $\delta_1$ | $Rel$ | $OPW$  |
  | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
  | [Midas](https://github.com/isl-org/MiDaS) | 0.651 | 0.288 | 0.676 | [DPT](https://github.com/isl-org/DPT) | 0.730 | 0.215 | 0.470 |
  | NVDS-Forward (Midas) | 0.700 | 0.240 | 0.207 |  NVDS-Forward (DPT) | 0.741 | 0.208 | 0.165 |
  | NVDS-Backward (Midas) | 0.699 | 0.240 | 0.218 |  NVDS-Backward (DPT) | 0.741 | 0.208 | 0.174 |
  | **NVDS-Final (Midas)** | **0.700** | **0.240** | **0.180** |  **NVDS-Final (DPT)** | **0.742** | **0.208** | **0.147** |
  

   
## 🍻 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=RaymondWang987/NVDS&type=Date)](https://star-history.com/#RaymondWang987/NVDS&Date)
    
   
  

  



## 🍭 Acknowledgement
We thank the authors for releasing [PyTorch](https://pytorch.org/), [MiDaS](https://github.com/intel-isl/MiDaS), [DPT](https://github.com/isl-org/DPT), [GMFlow](https://github.com/haofeixu/gmflow), [SegFormer](https://github.com/NVlabs/SegFormer), [VSS-CFFM](https://github.com/GuoleiSun/VSS-CFFM), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect), and [FFmpeg](http://ffmpeg.org/). Thanks for their solid contributions and cheers to the community.

## 📧 Citation
```
@InProceedings{Wang_2023_ICCV,
    author    = {Wang, Yiran and Shi, Min and Li, Jiaqi and Huang, Zihao and Cao, Zhiguo and Zhang, Jianming and Xian, Ke and Lin, Guosheng},
    title     = {Neural Video Depth Stabilizer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {9466-9476}
}
```
  


