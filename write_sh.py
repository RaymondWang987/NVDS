import os

all_videos = os.listdir('/xxx/vdw_test/')
all_videos.sort()

for i in range(len(all_videos)):

    with open('./test_VDW_NVDS_DPT.sh','a') as f:
        f.write('CUDA_VISIBLE_DEVICES=1 python infer_NVDS_dpt_bi.py'+ ' --video_dir /xxx/vdw_test/' + ' --base_dir /xxx/NVDS_VDW_Test/DPT/' + all_videos[i] +'/'  +' --vnum '+all_videos[i]+' --infer_w 896 --infer_h 384' +'\n')
        f.write('python vdw_test_metric.py'+ ' --gt_dir /xxx/vdw_test/' + ' --result_dir /xxx/NVDS_VDW_Test/DPT/'  +' --vnum '+all_videos[i]+'\n')


for i in range(len(all_videos)):

    with open('./test_VDW_NVDS_Midas.sh','a') as f:
        f.write('CUDA_VISIBLE_DEVICES=0 python infer_NVDS_midas_bi.py'+ ' --video_dir /xxx/vdw_test/' + ' --base_dir /xxx/NVDS_VDW_Test/Midas/' +all_videos[i] +'/'  +' --vnum '+all_videos[i]+  ' --infer_w 896 --infer_h 384' +'\n')
        f.write('python vdw_test_metric.py'+ ' --gt_dir /xxx/vdw_test/' + ' --result_dir /xxx/NVDS_VDW_Test/Midas/'  +' --vnum '+all_videos[i]+'\n')
