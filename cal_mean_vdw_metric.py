import numpy as np
import os
import math
import argparse


def get_args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dir', default='./NVDS_VDW_Test/Midas/', type=str)

    return parser
    

parser = get_args_parser()
args = parser.parse_args()

base_dir = args.test_dir + '/'
all_name = os.listdir(base_dir)
all_name.sort()
print(len(all_name))

count = 0

opw = 0
d1 = 0
d2 = 0
d3 = 0
rel = 0

opwdpt = 0
d1dpt = 0
d2dpt = 0
d3dpt= 0
reldpt = 0

opwf = 0
d1f = 0
d2f = 0
d3f= 0
relf = 0

opwb = 0
d1b = 0
d2b = 0
d3b= 0
relb = 0

for i in range(len(all_name)):

    count = count+1

    video_dir = base_dir + all_name[i]
    txt_opw = video_dir + '/result.txt'
    txt_depth = video_dir + '/accuracy.txt'

    all_line = []
    
    with open(txt_opw,'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip('\n')
            if line !='':
                all_line.append(line)
    
    opwdpt += float(all_line[1].split(',')[1][5:]) # initial
    opw += float(all_line[-1].split(',')[1][5:])   # bidirectional

    opwf += float(all_line[3].split(',')[1][5:]) # forward
    opwb += float(all_line[-3].split(',')[1][5:])   # backward
    
    
    
    
    

    all_line = []
    with open(txt_depth,'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip('\n')
            if line !='':
                all_line.append(line)
    
    d1 += float(all_line[-1].split(',')[0][3:])   # bidirectional
    d2 += float(all_line[-1].split(',')[1][3:])
    d3 += float(all_line[-1].split(',')[2][3:])
    rel += float(all_line[-1].split(',')[3][7:])

    d1b += float(all_line[-3].split(',')[0][3:])   # backward
    d2b += float(all_line[-3].split(',')[1][3:])
    d3b += float(all_line[-3].split(',')[2][3:])
    relb += float(all_line[-3].split(',')[3][7:])

    d1f += float(all_line[3].split(',')[0][3:])   # forward
    d2f += float(all_line[3].split(',')[1][3:])
    d3f += float(all_line[3].split(',')[2][3:])
    relf += float(all_line[3].split(',')[3][7:])    
    
    d1dpt += float(all_line[1].split(',')[0][3:])   # initial
    d2dpt += float(all_line[1].split(',')[1][3:])
    d3dpt += float(all_line[1].split(',')[2][3:])
    reldpt += float(all_line[1].split(',')[3][7:])


print('initial:',d1dpt/len(all_name),d2dpt/len(all_name),d3dpt/len(all_name),reldpt/len(all_name),opwdpt/len(all_name))
print('forward:',d1f/len(all_name),d2f/len(all_name),d3f/len(all_name),relf/len(all_name),opwf/len(all_name))
print('backward:',d1b/len(all_name),d2b/len(all_name),d3b/len(all_name),relb/len(all_name),opwb/len(all_name))
print('ours:',d1/len(all_name),d2/len(all_name),d3/len(all_name),rel/len(all_name),opw/len(all_name))
    
            
        


    