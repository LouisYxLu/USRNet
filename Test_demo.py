# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""


import torch
import torch.nn as nn

#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import time
import os
from USRNet import *
import utils_train

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir):    
	model_info = torch.load(checkpoint_dir + 'checkpoint_CDD.pth.tar')
	net = USRNet()
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()
	model.load_state_dict(model_info['state_dict'])
	optimizer = torch.optim.Adam(model.parameters())
	optimizer.load_state_dict(model_info['optimizer'])
	cur_epoch = model_info['epoch']

	return model

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './Input'#'./Input_Reside'
	result_dir = './Output'   #'./Output_Reside'    
	testfiles = os.listdir(test_dir)
	totalfiles = len(testfiles)
    
	print('> USRNet Testing, Total Images is %d < ...'%totalfiles)

	model = load_checkpoint(checkpoint_dir)

	for f in range(totalfiles):
		model.eval()
		with torch.no_grad():
			oimg = cv2.imread(test_dir + '/' + testfiles[f])/255.0      
			w,h,c= oimg.shape     
			img_l = hwc_to_chw(np.array(oimg).astype('float32'))
			input_var = torch.from_numpy(img_l.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
			s = time.time()
			E_out,_ = model(input_var,T=5)
			e = time.time()         
			print('FileName: %s'%testfiles[f] + ' ' + 'Shape: (%d,%d)'%(w,h) + ' ' +  'Time: %.4f'%(e-s) + ' ' +  'Done/Total: %d/%d'%((f+1),totalfiles))    
			
			E_out = chw_to_hwc(E_out.squeeze().cpu().detach().numpy())	               
			cv2.imwrite(result_dir + '/' + testfiles[f][:-4] + '_USRNet.png',np.clip(E_out*255,0.0,255.0))


                
	  
				
			
			

