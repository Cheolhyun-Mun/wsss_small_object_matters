#!/bin/bash
python main_ewc.py train \
    --config-path configs/voc12_deeplabv2.yaml \
    --margin 5 \
    --ewc_reg 500 \
    --ewc_iter 15000

python main_ewc.py test \
    --data_type TRUE \ # TRUE: PASCAL VOC / FALSE: PASCAL-B
    --config-path configs/voc12_deeplabv2.yaml \
    --margin 5 \
    --ewc_reg 500 \
    --ewc_iter 15000

python main_ewc.py crf \
    --data_type TRUE \ # TRUE: PASCAL VOC / FALSE: PASCAL-B
    --config-path configs/voc12_deeplabv2.yaml \
    --margin 5 \
    --ewc_reg 500 \
    --ewc_iter 15000