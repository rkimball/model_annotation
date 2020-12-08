#!/bin/bash
python3 tune_and_evaluate.py --log-dir="../logs" --target-os="linux" --target="llvm" --network="vgg_16_mxnet" --apply-log
#python3 tune_and_evaluate.py --log-dir="../logs" --target-os="linux" --target="llvm" --network="vgg_16_mxnet"