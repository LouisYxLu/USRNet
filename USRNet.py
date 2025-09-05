# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.nn as nn
import torch.nn.functional as F
import torch

class USRNet(nn.Module):
    def __init__(self):
        super(USRNet,self).__init__()

        self.mns = MainNetworkStructure(3,16)
         
    def forward(self,x,T):
        
        Fout,Eout = self.mns(x,T)
      
        return Fout, Eout 


class MainNetworkStructure(nn.Module):
    def __init__(self,inchannel,channel):
        super(MainNetworkStructure,self).__init__()

        self.en  = Encoder(channel)        
        self.de  = Decoder(channel)
        self.ede = Decoder(channel)#EdgeDecoder(channel)
        		        
        self.mid = NIL(channel)
        
        self.conv_in   = nn.Conv2d(3,channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_out1 = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)   
        self.conv_out2 = nn.Conv2d(channel,1,kernel_size=3,stride=1,padding=1,bias=False)  
        self.conv_e2r_8  = nn.Conv2d(8*channel,8*channel,kernel_size=3,stride=1,padding=1,bias=False)   
        self.conv_e2r_4  = nn.Conv2d(4*channel,4*channel,kernel_size=3,stride=1,padding=1,bias=False)      
                              
    def forward(self,x,T):
        
        x_in = self.conv_in(x)
        
        x_e11,x_e12,x_e21,x_e22,x_e31,x_e32,x_e41,x_e42 = self.en(x_in)
        
        xmout = self.mid(x_e41,T) 
           
        x_ede1,x_ede2,x_ede3,x_ede4 = self.ede(xmout,x_e42,x_e32,x_e22,x_e12)
        _,_,_,x_de4 = self.de(xmout,x_e41+x_ede1,x_e31+x_ede2,x_e21+x_ede3,x_e11+x_ede4)   

        x_out2 =  self.conv_out2(x_ede4)
                                
        x_out1 =  self.conv_out1(x_de4)

                    
        return x_out1,x_out2
    
class RB_E(nn.Module):    #Residual Block for Encoder (RB_E)
    def __init__(self,channel):                                
        super(RB_E,self).__init__()

        self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)     
        self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)      
        self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False) 
                
        self.conv_h = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)  
        self.conv_l = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False) 
                                                 
        self.act = nn.PReLU(channel)
        self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)

        self.lap = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).expand(channel, channel, -1, -1)       
        self.lap.weight = nn.Parameter(lap_kernel, requires_grad=False)

        self.gau = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        gau_kernel = torch.tensor([[1, 2, 1], [2, 8, 2], [1, 2, 1]], dtype=torch.float32).expand(channel, channel, -1, -1)       
        self.gau.weight = nn.Parameter(gau_kernel, requires_grad=False)
                           
    def forward(self,x):
        
        x_1 = self.act(self.norm(self.conv_1(x)))            
        
        x_lap = self.lap(x_1)
        x_high = self.conv_h(x_lap)
        x_low  = self.conv_l(x_1 - x_lap)#self.conv_l(self.gau())               

        x_2 = self.act(self.norm(self.conv_2(x_1 + x_high + x_low)))
                
        x_3 = self.act(self.norm(self.conv_3(x_2)) + x)        
                    
        return	x_3, x_high

class RB_D(nn.Module):    #MResidual Block for Decoder (RB_E)
    def __init__(self,channel):                                
        super(RB_D,self).__init__()

        self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)          
        self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)     
                
        self.act = nn.PReLU(channel)
        self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)
   
    def forward(self,x):
        
        x_1   = self.act(self.norm(self.conv_1(x)))
        x_2   = self.act(self.norm(self.conv_2(x_1)))
        x_3   = self.act(self.norm(self.conv_3(x_2)) + x)

        return	x_3
        		
class Encoder(nn.Module):
    def __init__(self,channel):
        super(Encoder,self).__init__()
        
        self.e1  = RB_E(channel)
        self.e2  = RB_E(channel*2)
        self.e3  = RB_E(channel*4)     
        self.e4  = RB_E(channel*8)   

        self.conv_e1te2 = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
        self.conv_e2te3 = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)  
        self.conv_e3te4 = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)   
        
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        
        e1out_1,e1out_2  = self.e1(x)        
        e2out_1,e2out_2  = self.e2(self.conv_e1te2(self.maxpool(e1out_1)))        
        e3out_1,e3out_2  = self.e3(self.conv_e2te3(self.maxpool(e2out_1)))
        e4out_1,e4out_2  = self.e4(self.conv_e3te4(self.maxpool(e3out_1)))        

        return e1out_1,e1out_2, e2out_1,e2out_2, e3out_1,e3out_2, e4out_1,e4out_2

    
class Decoder(nn.Module):
    def __init__(self,channel):
        super(Decoder,self).__init__()
        
        self.d1  = RB_D(channel*8)
        self.d2  = RB_D(channel*4)
        self.d3  = RB_D(channel*2)     
        self.d4  = RB_D(channel)   

        self.conv_d1td2 = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   
        self.conv_d2td3 = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)  
        self.conv_d3td4 = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)   
     

    def _upsample(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')

    def forward(self,x,e4,e3,e2,e1):
        
        d1out  = self.d1(x + e4)
        d2out  = self.d2(self._upsample(self.conv_d1td2(d1out),e3) + e3)
        d3out  = self.d3(self._upsample(self.conv_d2td3(d2out),e2) + e2)     
        d4out  = self.d4(self._upsample(self.conv_d3td4(d3out),e1) + e1)  
        
        return d1out,d2out,d3out,d4out 
        
class NIL(nn.Module):    #Nodes Independent Learning and Inferring
    def __init__(self,channel):
        super(NIL,self).__init__()

        self.m0 = NILB(channel*8)        
        self.m1 = NILB(channel*8)
        self.m2 = NILB(channel*8)
        self.m3 = NILB(channel*8)
		
    def forward(self,x,T,istraining = True):

        if istraining is True:        
            if int(T) == 0:      #Haze
                xout = self.m0(x)            
            elif int(T) == 1:    #Rain
                xout = self.m1(x)            
            elif int(T) == 2:    #Snow
                xout = self.m2(x)                        
            elif int(T) == 3:    #Rain + Snow
                xout = self.m2(self.m1(x)) + self.m1(self.m2(x))            
            elif int(T) == 4:    #Haze + Rain
                xout = self.m1(self.m0(x))            
            elif int(T) == 5:    #Haze + Snow
                xout = self.m2(self.m0(x))            
            elif int(T) == 6:    #Haze + Rain + Snow			
                x_haze = self.m0(x)
                x_snowrainhaze = self.m2(self.m1(x_haze))
                x_rainsnowhaze = self.m1(self.m2(x_haze))				
                xout = self.m3(x_snowrainhaze + x_rainsnowhaze)
        else:     
                x_haze = self.m0(x)
                x_snowrainhaze = self.m2(self.m1(x_haze))
                x_rainsnowhaze = self.m1(self.m2(x_haze))				
                xout = self.m3(x_snowrainhaze + x_rainsnowhaze)        		       
        return xout

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.LayerNorm([in_channels // reduction, 1, 1]),
            nn.PReLU(in_channels // reduction),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        )

    def forward(self, x):
        batch, channel, height, width = x.size()

        # Context Modeling
        input_x = x
        context_mask = self.conv_mask(input_x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.view(batch, 1, height, width)

        context = input_x * context_mask
        context = torch.sum(context, dim=(2, 3), keepdim=True)

        # Transform
        channel_add_term = self.channel_add_conv(context)

        out = x + channel_add_term

        return out


class NILB(nn.Module):  # Nodes Independent Learning Block
    def __init__(self, channel):
        super(NILB, self).__init__()

        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.gc_block_1 = GlobalContextBlock(channel)

        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.gc_block_2 = GlobalContextBlock(channel)

        self.conv_3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.act = nn.PReLU(channel)
        self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)

    def forward(self, x):
        x_1 = self.act(self.norm(self.conv_1(x)))
        x_1 = self.gc_block_1(x_1)  # Apply GCNet after the first convolution

        x_2 = self.act(self.norm(self.conv_2(x_1)))
        x_2 = self.gc_block_2(x_2)  # Apply GCNet after the second set of convolutions

        x_out = self.act(self.norm(self.conv_3(x_2)) + x)

        return x_out

