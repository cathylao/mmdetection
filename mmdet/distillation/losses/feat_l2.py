import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class Feat_L2(nn.Module):
    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 weight=1.0,
                 ):
        super(Feat_L2, self).__init__()
        self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        self.weight = weight

    def forward(self,
                feat_S,
                feat_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert feat_S.shape[-2:] == feat_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            feat_S = self.align(feat_S)
        loss = F.mse_loss(feat_S,feat_T)*self.weight
        return loss

    # def last_zero_init(self, m):
    #     if isinstance(m, nn.Sequential):
    #         constant_init(m[-1], val=0)
    #     else:
    #         constant_init(m, val=0)

    
    # def reset_parameters(self):
    #     kaiming_init(self.conv_mask_s, mode='fan_in')
    #     kaiming_init(self.conv_mask_t, mode='fan_in')
    #     self.conv_mask_s.inited = True
    #     self.conv_mask_t.inited = True

    #     self.last_zero_init(self.channel_add_conv_s)
    #     self.last_zero_init(self.channel_add_conv_t)