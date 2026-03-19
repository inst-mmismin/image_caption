python step1.py --epochs 20 \
                --learning_rate 0.01 \
                --min_lr 0.0001 \
                --batch_size 512 \
                --use_contrastive \
                --proj_type linear \
                --use_layer_norm 

python step1.py --epochs 20 \
                --learning_rate 0.01 \
                --min_lr 0.0001 \
                --batch_size 512 \
                --use_contrastive \
                --proj_type mlp \
                --use_layer_norm 

python step1.py --epochs 20 \
                --learning_rate 0.01 \
                --min_lr 0.0001 \
                --batch_size 512 \
                --use_contrastive \
                --weight_contrastive 1 \
                --proj_type mlp \
                --use_layer_norm 

python step1.py --epochs 20 \
                --learning_rate 0.01 \
                --min_lr 0.0001 \
                --batch_size 512 \
                --use_contrastive \
                --contra_temp 0.05 \
                --proj_type linear \
                --use_layer_norm 


python step1.py --epochs 20 \
                --learning_rate 0.01 \
                --min_lr 0.0001 \
                --batch_size 512 \
                --use_contrastive \
                --weight_contrastive 1 \
                --contra_temp 0.05 \
                --proj_type linear \
                --use_layer_norm 

