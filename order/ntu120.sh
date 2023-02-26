CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh configs/sscls/ntu120xsub_5parts_kp.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh configs/sscls/ntu120xsub_5parts_lb.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh configs/sscls/ntu120xsub_10parts_kplb.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh configs/sscls/ntu120xsub_17parts_kp.py 4 --validate --seed 42 --deterministic
