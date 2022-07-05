import torch.nn.functional as F
import torch
import numpy as np
from distutils.version import LooseVersion
import cv2

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def CELoss(input, target, margin, give_weight = True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    batch_target_mask = target.detach()

    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=None, ignore_index=255, reduction='none')
       

    if give_weight:
        loss = loss.view(n,h,w)
        weight = np.ones((n,h,w))
        for idx in range(n):
            target_mask = batch_target_mask[idx].cpu().numpy().copy()
            whole_pix = np.sum((target_mask<255)&(target_mask>0))
            for cl in np.unique(target_mask):
                if cl == 0 or cl == 255:
                    continue
                target_mask_class = np.zeros((h,w))
                target_mask_class[target_mask==cl]=1
                contours = cv2.findContours(
                            image=target_mask_class.astype('uint8'),
                            mode=cv2.RETR_TREE,
                            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
                        
                if len(contours)==0:
                    continue
                    
                for contour in contours:
                    contour_mask = np.zeros((h,w))
                    cv2.drawContours(contour_mask,[contour],0,1,-1)
                    num_pix = np.sum(contour_mask)

                    weight[idx][contour_mask>0] = min(margin,(whole_pix/num_pix))
        
        weight = torch.from_numpy(weight).cuda()
        loss *= weight
        weight = weight[loss!=0]
        loss = loss.view(-1)
        loss = torch.sum(loss[loss!=0])/torch.sum(weight.view(-1))
    else:
        loss = loss[loss!=0].mean()

    
    return loss