import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as io
import os
import argparse
import glob
import numpy as np
import argparse
#vnum = 'study_0002'

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vnum",
        default= '000423',
        type=str
    )
    parser.add_argument(
        "--infer_w",
        default= '896',
        type=int
    )
    parser.add_argument(
        "--infer_h",
        default= '384',
        type=int
    )

    return parser

parser = get_args_parser()
args = parser.parse_args()
vnum = args.vnum #'000010'




dir_rgb = './demo_videos/'+vnum+'/left/'
rgb = os.listdir(dir_rgb)
rgb.sort() # key= lambda x:int(x[:-4]))
#rgb = rgb[len(rgb)//2:]




dir_dpt = './demo_outputs/dpt_init/'+vnum+'/initial/color/'
dpt = os.listdir(dir_dpt)
dpt.sort(key= lambda x:int(x[-10:-4]))

dir_dptours = './demo_outputs/dpt_init/'+vnum+'/mix/color/'
dptours = os.listdir(dir_dptours)
dptours.sort(key= lambda x:int(x[-10:-4]))

dir_midas = './demo_outputs/midas_init/'+vnum+'/initial/color/'
midas = os.listdir(dir_midas)
midas.sort(key= lambda x:int(x[-10:-4]))

dir_midasours = './demo_outputs/midas_init/'+vnum+'/mix/color/'
video_dir = './demo_outputs_videos/compare_'+vnum+'.avi'#'/data1/wangyiran/mytrans/plt/2048.avi'#'/data1/wangyiran/mytrans/firstmae/asee/auau.avi'
# set saved fps
fps = 24 
# get frames list
midasours = os.listdir(dir_midasours)
midasours.sort(key= lambda x:int(x[-10:-4]))

img_size = (int(args.infer_w),int(args.infer_h))
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, (img_size[0]*2,img_size[1]*3))


for j in range(len(midasours)):
        
        print('frame:',j+1,'/',len(midasours))

        rgb_path = os.path.join(dir_rgb, rgb[j])
        rgbf = cv2.imread(rgb_path)
        rgbf = cv2.resize(rgbf,img_size)
        rgbf = np.hstack((rgbf, rgbf))

        dpt_path = os.path.join(dir_dpt, dpt[j])
        dptf = cv2.imread(dpt_path)
        dptf = cv2.resize(dptf,img_size)
        midas_path = os.path.join(dir_midas, midas[j])
        midasf = cv2.imread(midas_path)
        midasf = cv2.resize(midasf,img_size)
        singlef = np.hstack((dptf,midasf))

        dptours_path = os.path.join(dir_dptours, dptours[j])
        dptoursf = cv2.imread(dptours_path)
        dptoursf = cv2.resize(dptoursf,img_size)
        midasours_path = os.path.join(dir_midasours, midas[j])
        midasoursf = cv2.imread(midasours_path)
        midasoursf = cv2.resize(midasoursf,img_size)
        oursf = np.hstack((dptoursf,midasoursf))

        rgb_single = np.vstack((rgbf,singlef))
        rgb_single_ours = np.vstack((rgb_single,oursf))

        #print(rgb_single_ours.shape,(384*3,896*2))
        #exit(1)
        videowriter.write(rgb_single_ours)


videowriter.release()

        

