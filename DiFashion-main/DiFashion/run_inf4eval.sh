nohup accelerate launch --config_file config.yaml inf4eval.py --eta 0.1 --task $1 --mode $2 --use_ema --use_ema_fashion --mixed_precision fp16 --center_crop --cache_item_latents --inference_only --test_batch_size 1 --allow_tf32 >log_inf4eval_$1_$2.txt 2>&1 &

# sh run_inf4eval.sh FITB test
# sh run_inf4eval.sh GOR test
