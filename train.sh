export PYTHONPATH=`pwd`:$PYTHONPATH
CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python tools/train.py --config-name config.yaml 
CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=0 python tools/test.py --config-name config.yaml

python tools/visualize.py --config-name config.yaml