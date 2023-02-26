CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp0_t9/t9nsnr_5p_lr30.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp0_t9/t9nsnr_5p_lr3.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp0_t9/t9nsnr_5p_lr1.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp0_t9/t9nsnr_5p_lrd4.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp0_t9/t9nsnr_5p_lrd1.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp0_t9/t9nsnr_5p_lrd01.py 4 --validate --seed 42 --deterministic
