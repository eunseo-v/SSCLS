# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp4/t9nsnr_wfwc_5p_lb.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp4/t9nsnr_wfwc_10p_kplb.py 4 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp4/t3nsnr_wfwc.py 4 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp4/t5nsnr_wfwc.py 4 --validate --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp4/t7nsnr_wfwc.py 4 --validate --seed 42 --deterministic
