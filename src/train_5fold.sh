device='5'
save_dir='../save_dir/bigger_tp99/'

# train 5fold
for fold_idx in {0..4}
do  
    echo === fold $fold_idx ===

    # train 49
    python3 train.py \
        -c=config.json \
        -d=$device \
        --fold_idx=$fold_idx \
        --save_dir=$save_dir/fold$fold_idx
done

python3 test.py \
    --config=$save_dir/fold0/config.json \
    --output_dir=$save_dir \
    --device=$device
