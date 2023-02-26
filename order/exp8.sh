CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp8/t7_nsnr_wfwc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp8/t5_nsnr_wfwc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp8/t3_nsnr_wfwc_ft.py 4 --validate --seed 42 --deterministic
