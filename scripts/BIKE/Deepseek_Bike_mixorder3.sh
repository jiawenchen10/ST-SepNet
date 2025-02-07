model_name=pool
train_epochs=50
learning_rate=0.0002
llama_layers=6

master_port=0
num_process=2
batch_size=32
d_model=16
d_ff=32
comment='LLAMA8b-Bike'
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
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model 'deepseek2b' \
  --llm_dim 1536 \
  --gamma  0.0   \
  --fusion_gate  'adaptive' \
  --enc_in    $node_num  \
  --dec_in   $node_num  \
  --c_out  $node_num  \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --scale_hyperedges 3 \
  >>./result//LLAMA8b_BIKEIN_48_48_order3.txt


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
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model 'deepseek2b' \
  --llm_dim 1536 \
  --gamma  0.0   \
  --fusion_gate  'adaptive' \
  --enc_in    $node_num  \
  --dec_in   $node_num  \
  --c_out  $node_num  \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --scale_hyperedges 3 \
  >>./result//LLAMA8b_BIKEOUT_48_48_order3.txt
