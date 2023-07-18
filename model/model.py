import torch
import sys
import torch.nn as nn
from einops import rearrange
from model.module.transformer import Transformer as Transformer_encoder
from model.module.posegroup_transformer import Transformer as Group_Transformer
from model.module.hypothesis_transformer import Transformer as Hypothesis_transformer


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.embedding = nn.Sequential(nn.Conv1d(args.in_channels*args.n_joints, args.channel*args.n_joints, kernel_size=1))    
        
        self.Encoder_S = Transformer_encoder(depth=args.layers, 
                                             embed_dim=args.channel,
                                             mlp_hidden_dim=args.channel*3, 
                                             h=8, 
                                             drop_rate=0.1,
                                             frame_length=args.frames, 
                                             temp_or_spat='spatial', 
                                             joint_num=17)
       
        self.Encoder_T = Transformer_encoder(depth=args.layers, 
                                             embed_dim=args.channel*args.n_joints,
                                             mlp_hidden_dim=args.channel*args.n_joints*3, 
                                             h=8, 
                                             drop_rate=0.1,
                                             frame_length=args.frames, 
                                             temp_or_spat='temporal', 
                                             joint_num=17)
           
        self.Encoder_G = Group_Transformer(depth=3, 
                                           args_channel=args.channel,
                                           embed_dim=args.channel*args.n_joints, 
                                           mlp_hidden_dim=args.channel*args.n_joints*3,
                                           h=8, 
                                           drop_rate=0.1,
                                           frame_length=args.frames) 
        
        self.fusion_transformer = Hypothesis_transformer(depth=3, 
                                                         embed_dim=args.channel*args.n_joints, 
                                                         mlp_hidden_dim=args.channel*args.n_joints*2, 
                                                         length=args.frames)

        
        self.fcn_base = nn.Sequential(
            nn.Linear(args.channel*args.n_joints*3, args.channel*args.n_joints*2),
            nn.BatchNorm1d(args.channel*args.n_joints*2, momentum=0.1),
            nn.Linear(args.channel*args.n_joints*2, args.channel*args.n_joints),
            nn.BatchNorm1d(args.channel*args.n_joints, momentum=0.1),
            nn.Linear(args.channel*args.n_joints, args.out_channels*args.n_joints)) 

        self.Refinement = nn.Sequential(
            nn.Conv1d(args.channel*args.n_joints*3, args.channel*args.n_joints*2, kernel_size=3, stride=3), 
            nn.BatchNorm1d(args.channel*args.n_joints*2, momentum=0.1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(args.channel*args.n_joints*2, args.channel*args.n_joints*2, kernel_size=3, stride=3), 
            nn.BatchNorm1d(args.channel*args.n_joints*2, momentum=0.1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(args.channel*args.n_joints*2, args.channel*args.n_joints, kernel_size=9, stride=3),
            nn.BatchNorm1d(args.channel*args.n_joints, momentum=0.1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(args.channel*args.n_joints, args.out_channels*args.n_joints, kernel_size=1, stride=1))


    def forward(self, x, args):  
        B, F, J, C = x.shape # x.shape : [B, F, J, 2]
          
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()  
        x = self.embedding(x)  # x : [B, J x C, F] = x : [B, J x 2, F]    
        x = rearrange(x, 'b (j c) f -> b f (j c)', j=J).contiguous()  # o_x.shape : [B, F, J x C]

        ## all joints temporal 
        x1 = self.Encoder_T(x)  # x1.shape : [B, F, J x C]

        ## pose grammar
        # kinematics grammar
        x21 = x1[:, :, 0 : 5 * args.channel]               # x21.shape : [B, F, 5 x C]
        x22 = x1[:, :, 5 * args.channel : 11 * args.channel] # x22.shape   : [B, F, 6 x C]
        x23 = x1[:, :, 11 * args.channel :]                  # x23.shape   : [B, F, 6 x C]
        x2 = self.Encoder_G(x21, x22, x23) # x2.shape : [B, F, 17 x C]
        x2 = rearrange(x2, 'b f (j c) -> (b f) j c', j=J).contiguous()  
    
        ## spatial correlation
        x_S = self.Encoder_S(x2)  # x_S.shape : [B x F, J, C] = 
        x_S = x_S.permute(0, 2, 1).contiguous()  # x_S.shape : [B x F, C, J] = 
        x_S = rearrange(x_S, '(b f) c j -> b f (j c)', b=B).contiguous()  # x_S.shape : [B, F, J, C] = 
        
        
        x2 = rearrange(x2, '(b f) j c -> b f (j c)', b=B).contiguous() # x2.shape : [B, F, J x C] = 
        output = self.fusion_transformer(x1, x2, x_S) # output.shape : [B x F, J x C x 3] =       
        output = rearrange(output, 'b f C -> (b f) C', b=B).contiguous() # output.shape : [B x F, J x C x 3] = 
        
        # Regression and get output_base      
        output_base = self.fcn_base(output)  # output_base.shape : [B x F, J x 3] = output.shape : [B x F, J x C x 3]  
        output_base = rearrange(output_base, '(b f) (j c) -> b f j c', b=B, j=J).contiguous()
        
        # compute output_refine
        output = rearrange(output, '(b f) C -> b C f', b=B).contiguous() # output.shape : [B, J x C x 3, F] = 
        output_refine = self.Refinement(output) # output_refine.shape : [B, 17 x 3, 1] = 
        output_refine = rearrange(output_refine, 'b (j c) f -> b f j c', j=J).contiguous()  # output output_refine : [B, 1, 17, 3]
        
        return output_base, output_refine






