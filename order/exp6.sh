CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp6/t9nsnr_nfnc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp6/t9nswr_nfnc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp6/t9wsnr_nfnc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp6/t9wswr_nfnc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp6/t9wswr_nfwc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp6/t9wswr_wfnc_ft.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp6/t9wswr_wfwc_ft.py 4 --validate --seed 42 --deterministic
