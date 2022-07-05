# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import cv2

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def get_label_for_instance(label_instance, label_semantic, pred, idx):
    pred_mask = np.ones(pred.shape)
    label_semantic_mask = np.ones(label_semantic.shape)
    label_semantic_others_mask = np.ones(label_semantic.shape)
    if idx == 0:
        cls_num = idx
        label_semantic_mask[label_semantic>0] = 0
        pred_mask[pred!=0] = 0

        return label_semantic_others_mask, label_semantic_mask, pred_mask, cls_num
    else:
        label_semantic_mask[label_instance!=idx] = 0
        cls_num = np.max(label_semantic_mask*label_semantic).astype(int)
        pred_mask[pred!=cls_num] = 0

        label_semantic_others_mask[label_instance==idx] = 0
        label_semantic_others_mask[label_semantic!=cls_num] = 0

        return label_semantic_others_mask, label_semantic_mask, \
                pred_mask, cls_num

def get_instance_num(label_instance):
    number = np.unique(label_instance)
    if number[-1] != 255:
        return number[-1]
    return number[-2] 

def check_nn(contour_mask, label_true):
    whole_area = label_true.astype(int) + contour_mask
    number, num_count = np.unique(whole_area,return_counts=True)
    if number[-1]!=2:
        return 0
    num_overlapped_pix = num_count[-1]
    
    return num_overlapped_pix
    

def IAmIoU(label_instance_batch, label_semantic_batch, pred_batch, n_class):
    iu_stack = np.zeros(n_class)
    class_stack = np.zeros(n_class)
    iu_small_stack = np.zeros(n_class)
    class_small_stack = np.zeros(n_class)
    iu_medium_stack = np.zeros(n_class)
    class_medium_stack = np.zeros(n_class)
    iu_large_stack = np.zeros(n_class)
    class_large_stack = np.zeros(n_class)
    instance_size = 'l'
    
    for label_semantic, label_instance, pred in zip(label_semantic_batch, label_instance_batch, pred_batch):
        max_instance_num = get_instance_num(label_instance)

        for idx in range(0, max_instance_num+1):
            seg_temp_other, seg_temp, preds_temp, cls_num = \
                get_label_for_instance(label_instance.copy(), label_semantic.copy(), \
                                        pred.copy(), idx)
            
            size = np.sum(seg_temp)
            if size < 32*32:
                instance_size = 's'
            elif size < 96*96:
                instance_size = 'm'
            else:
                instance_size = 'l'

            real_area = (label_instance < 255)
            union, pred_pix, intersection = 0, 0, 0

            if idx == 0 : # background
                if np.sum(seg_temp[real_area]) == 0:
                    continue
                whole_area = seg_temp[real_area]+preds_temp[real_area]
                number, num_count = np.unique(whole_area,return_counts=True)
                if number[-1]!=2:
                    class_stack[cls_num]+=1
                    continue
                union = np.sum(preds_temp[real_area])+np.sum(seg_temp[real_area])-num_count[-1]
                intersection = num_count[-1]
                
            else :
                accumul_mask = np.ones(preds_temp.shape) 
                contours = cv2.findContours(
                    image=preds_temp.astype('uint8'),
                    mode=cv2.RETR_TREE,
                    method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
                    
                if len(contours)==0:
                    class_stack[cls_num] += 1
                    if instance_size == 's':
                        class_small_stack[cls_num] += 1
                    elif instance_size == 'm':
                        class_medium_stack[cls_num] += 1
                    else:
                        class_large_stack[cls_num] += 1
                    continue
                
                for contour in contours:
                    contour_mask = np.zeros(pred.shape)
                    cv2.drawContours(contour_mask,[contour],0,1,-1)
                    contour_mask *= (preds_temp*accumul_mask)
                    accumul_mask -= contour_mask
                    whole_area = contour_mask[real_area]+seg_temp[real_area]
                    number, num_count = np.unique(whole_area,return_counts=True)
                    if number[-1]!=2:
                        continue

                    num_overlapped_pix = check_nn(contour_mask[real_area], seg_temp_other[real_area])
                    if num_overlapped_pix == 0:
                        pred_pix += np.sum(contour_mask[real_area])
                    else:
                        nn_rate = (num_count[-1]/(num_count[-1]+num_overlapped_pix))
                        nn_rate *= (np.sum(contour_mask[real_area])-num_count[-1]-num_overlapped_pix)
                        pred_pix += (int(nn_rate)+num_count[-1])
                    union -= num_count[-1]
                    intersection += num_count[-1]
                    
                union += (pred_pix+np.sum(seg_temp[real_area]))
            
                if instance_size == 's':
                    iu_small_stack[cls_num] += (intersection/union)
                    class_small_stack[cls_num] += 1
                elif instance_size == 'm':
                    iu_medium_stack[cls_num] += (intersection/union)
                    class_medium_stack[cls_num] += 1
                else:
                    iu_large_stack[cls_num] += (intersection/union)
                    class_large_stack[cls_num] += 1

            iu_stack[cls_num] += (intersection/union)
            class_stack[cls_num] += 1
            
    return {
         "IoU stack": iu_stack,
         "Class stack": class_stack,
         "Small IoU stack": iu_small_stack,
         "Small Class stack": class_small_stack,
         "Medium IoU stack": iu_medium_stack,
         "Medium Class stack": class_medium_stack,
         "Large IoU stack": iu_large_stack,
         "Large Class stack": class_large_stack,
     }

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    )
    hist = hist.reshape(n_class, n_class)
    return hist

def mIoU(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return {
        "hist": hist
    }