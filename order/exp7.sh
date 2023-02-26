# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp7/ncrc_adamw5e-4.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,4 ./tools/dist_train.sh configs/sscls/exp7/ncrc_adamw5e-4_lb.py 4 --validate --seed 42 --deterministic
