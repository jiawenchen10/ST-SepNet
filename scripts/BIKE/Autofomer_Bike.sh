model_name=Autoformer
train_epochs=50
learning_rate=0.0005
llama_layers=32

master_port=
num_process=2
batch_size=16
d_model=16
d_ff=32
comment='Autoformer-Bike'
accelerate launch   --mixed_precision bf16  --dynamo_backend 'no' --num_processes $num_process  run_main.py \
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
  >>./result/Autoformer_BIKEIN_48_48.txt


accelerate launch   --mixed_precision bf16  --dynamo_backend 'no' --num_processes 2  run_main.py \
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
  >>./result/Autoformer_BIKEOUT_48_48.txt


