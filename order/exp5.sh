CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp5/t9nsnr_5p_ft_lr1.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp5/t9nsnr_5p_ft_lrd1.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp5/t9nsnr_5p_ft_lrd01.py 4 --validate --seed 42 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/sscls/exp5/t9nsnr_5p_ft_lrd01_nopre.py 4 --validate --seed 42 --deterministic
