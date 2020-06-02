# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

# change -np and localhost: to number of gpus

cd /deep-learning-models/models/vision/detection
export PYTHONPATH=${PYTHONPATH}:${PWD}

mpirun -np 8 \
--H localhost:8 \
--allow-run-as-root \
-mca btl_tcp_if_exclude lo,docker0 \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
python tools/train_backbone.py \
--train_data_dir /deep-learning-models/models/vision/detection/data/imagenet/train \
--validation_data_dir /deep-learning-models/models/vision/detection/data/imagenet/validation \
--batch_size 256 


