import os
import numpy as np
import re
import pdb

param_dir ='/home/alexandracarlson/Desktop/FCN8_test_results/out-gtaSim10k2city-STgen-4convlayer-fcnstyleloss-voc-4epoch'
#
dataset_slug = 'gta_sim10k_voc_crop'
param_file_loc = os.path.join(param_dir,'checkpoints',dataset_slug,'aug_imgs')
#pdb.set_trace()
files = [os.path.join(param_file_loc,fn) for fn in os.listdir(param_file_loc) if '.txt' in fn]

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

keyparams = ['chromab','blur','exposure','noise','color']
chromab_list=[]
blur_list=[]
exposure_list=[]
noise_list=[]
color_list=[]

for f in files:
	## open file
	gf = open(f,'r')
	filestr = gf.read()
	gf.close()
	# parse string
	params_start_inds = [filestr.find(param) for param in keyparams]
	params_end_inds = [ filestr[startind:].find(']')+startind for startind in params_start_inds]
	params_inds = zip(params_start_inds, params_end_inds)
	params_strs = [filestr[start:end].split('[')[1] for start,end in params_inds]
	params_list = [param.split(' ') for param in params_strs]
	pp = [[float(oo.strip()) for oo in p if oo] for p in params_list ]
	chromab_list.append(pp[0])
	blur_list.append(pp[1])
	exposure_list.append(pp[2])
	noise_list.append(pp[3])
	color_list.append(pp[4])

print 'chromab param min: {0}, {1}, {2}, {3}, {4}, {5}, {6}'.format(*[str(x) for x in np.min(np.array(chromab_list),axis=0)])
print 'chromab param max: {0}, {1}, {2}, {3}, {4}, {5}, {6}'.format(*[str(x) for x in np.max(np.array(chromab_list),axis=0)])
print 'chromab mean {0}, {1}, {2}, {3}, {4}, {5}, {6}'.format( *[ str(x) for x in np.mean(np.array(chromab_list),axis=0)] )
print 'chromab stddev {0}, {1}, {2}, {3}, {4}, {5}, {6}'.format( *[str(x) for x in np.std(np.array(chromab_list),axis=0)] )

print 'blur param min: {0}, {1}'.format(*[str(x) for x in np.min(np.array(blur_list),axis=0)])
print 'blur param max: {0}, {1}'.format(*[str(x) for x in np.max(np.array(blur_list),axis=0)])
print 'blur parammean {0}, {1}'.format( *[ str(x) for x in np.mean(np.array(blur_list),axis=0)] )
print 'blur param stddev {0}, {1}'.format( *[str(x) for x in np.std(np.array(blur_list),axis=0)] )

print 'exposure param min: {0}, {1}'.format(*[str(x) for x in np.min(np.array(exposure_list),axis=0)])
print 'exposure param max: {0}, {1}'.format(*[str(x) for x in np.max(np.array(exposure_list),axis=0)])
print 'exposure param mean {0}, {1}'.format( *[ str(x) for x in np.mean(np.array(exposure_list),axis=0)] )
print 'exposure param stddev {0}, {1}'.format( *[str(x) for x in np.std(np.array(exposure_list),axis=0)] )

print 'noise param min: {0}, {1}, {2}, {3}, {4}, {5}'.format(*[str(x) for x in np.min(np.array(noise_list),axis=0)])
print 'noise param max: {0}, {1}, {2}, {3}, {4}, {5}'.format(*[str(x) for x in np.max(np.array(noise_list),axis=0)])
print 'nose param mean {0}, {1}, {2}, {3}, {4}, {5}'.format( *[ str(x) for x in np.mean(np.array(noise_list),axis=0)] )
print 'noise param stddev {0}, {1}, {2}, {3}, {4}, {5}'.format( *[str(x) for x in np.std(np.array(noise_list),axis=0)] )

print 'color param min: {0}, {1}'.format(*[str(x) for x in np.min(np.array(color_list),axis=0)])
print 'color param max: {0}, {1}'.format(*[str(x) for x in np.max(np.array(color_list),axis=0)])
print 'color param mean {0}, {1}'.format( *[ str(x) for x in np.mean(np.array(color_list),axis=0)] )
print 'color param stddev {0}, {1}'.format( *[str(x) for x in np.std(np.array(color_list),axis=0)] )

