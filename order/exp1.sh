CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp1/t9nsnr_nfnc.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp1/t9nsnr_nfwc.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp1/t9nsnr_wfnc.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp1/t9nsnr_wfwc.py 4 --validate --seed 42 --deterministic
