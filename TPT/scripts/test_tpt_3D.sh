#!/bin/bash

data_root='/hub_data2/jisoo/AAA534/data/modelnet40_ply_hdf5_2048/'
testsets=ModelNet40
arch=ViT-B/16
bs=1
ctx_init=A_gray,_unclear_3D_model_of

CUDA_VISIBLE_DEVICES=1 python ./tpt_classification_3D.py \
    ${data_root} \
    --test_sets ${testsets} \
    -a ${arch} \
    -b ${bs} \
    --gpu 0 \
    --tpt \
    --n_ctx 4 \
    --ctx_init ${ctx_init} \
    -p 500 \
    --selection_p 0.6