#!/usr/bin/env bash
set -e

python train.py \
  --config configs/datasets/cifar10/cifar10.yml \
  configs/networks/resnet18_32x32.yml \
  configs/pipelines/train/train_sem.yml \
  configs/preprocessors/base_preprocessor.yml \
  --num_workers 12 \
  --network.checkpoint ./results/pre-train/best.ckpt

python test.py \
  --config configs/datasets/cifar10/cifar10.yml \
  configs/datasets/cifar10/cifar10_fsood.yml \
  configs/networks/resnet18_32x32.yml \
  configs/pipelines/test/test_fsood.yml \
  configs/preprocessors/base_preprocessor.yml \
  configs/postprocessors/gmm.yml \
  --num_workers 8 \
  --mark 0
