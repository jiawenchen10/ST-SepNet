model_name=DLinear
train_epochs=50
learning_rate=0.0005
llama_layers=32

master_port=8879
num_process=1
batch_size=16
d_model=16
d_ff=32
comment='DLinear-Bike'
accelerate launch   --mixed_precision bf16  --dynamo_backend 'no' --num_processes $num_process   --main_process_port $master_port  run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Bike/ \
  --data_path inflow.csv \
  --model_id BIKE_48_48 \
  --model $model_name \
  --data inflow \
  --features M \
  --seq_len 48 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in 295 \
  --dec_in 295 \
  --c_out 295 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  >>./result//DLinear_BIKEIN_48_48.txt


accelerate launch   --mixed_precision bf16  --dynamo_backend 'no' --num_processes 1    --main_process_port $master_port   run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Bike/ \
  --data_path outflow.csv \
  --model_id BIKEOUT_48 \
  --model $model_name \
  --data outflow \
  --features M \
  --seq_len 48 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in 295 \
  --dec_in 295 \
  --c_out 295 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  >>./result//DLinear_BIKEOUT_48_48.txt


# accelerate launch --multi_gpu --mixed_precision bf16 --dynamo_backend 'no' --num_machines 1 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 336 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --lradj 'COS'\
#   --learning_rate 0.001 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 720 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment