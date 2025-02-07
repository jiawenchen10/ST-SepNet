
model_name=pool
train_epochs=30
learning_rate=0.0005
llama_layers=8
master_port=0
num_process=2
batch_size=32
d_model=16
d_ff=32
node_num=358
comment='LLAMA7b-PEMS03'
accelerate launch   --mixed_precision bf16  --dynamo_backend 'no' --num_processes $num_process  run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS03/ \
  --data_path PEMS03.csv \
  --model_id PEMS03_48_48 \
  --adjacency_path PEMS03_adj.csv  \
  --model $model_name \
  --data PEMS03 \
  --features M \
  --seq_len 48 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in    $node_num  \
  --dec_in   $node_num  \
  --c_out  $node_num  \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model 'LLAMA7b' \
  --llm_dim 4096 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --node_num   $node_num  \
  --gamma  1.0   \
  --scale_hyperedges 3 \
  --gcn_true True \
  --hgcn_true False \
  --hgat_true False \
  >>./result/PEMS03_hypergnn/LLAMA7b_PEMS03_48_48.txt




model_name=pool
train_epochs=30
learning_rate=0.0005
llama_layers=8
master_port=0
num_process=2
batch_size=32
d_model=16
d_ff=32
node_num=358
comment='LLAMA7b-PEMS03'
accelerate launch   --mixed_precision bf16  --dynamo_backend 'no' --num_processes $num_process  run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS03/ \
  --data_path PEMS03.csv \
  --model_id PEMS03_48_48 \
  --adjacency_path PEMS03_adj.csv  \
  --model $model_name \
  --data PEMS03 \
  --features M \
  --seq_len 48 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in    $node_num  \
  --dec_in   $node_num  \
  --c_out  $node_num  \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model 'LLAMA7b' \
  --llm_dim 4096 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --node_num   $node_num  \
  --gamma  0.5  \
  --scale_hyperedges 3 \
  --gcn_true True \
  --hgcn_true True \
  --hgat_true False \
  >>./result/PEMS03_hypergnn/LLAMA7b_PEMS03_48_48_mixorder.txt



node_num=358
comment='LLAMA7b-PEMS03'
accelerate launch   --mixed_precision bf16  --dynamo_backend 'no' --num_processes $num_process  run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS03/ \
  --data_path PEMS03.csv \
  --model_id PEMS03_48_48 \
  --adjacency_path PEMS03_adj.csv  \
  --model $model_name \
  --data PEMS03 \
  --features M \
  --seq_len 48 \
  --label_len 48 \
  --pred_len 48 \
  --factor 3 \
  --enc_in    $node_num  \
  --dec_in   $node_num  \
  --c_out  $node_num  \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model 'LLAMA7b' \
  --llm_dim 4096 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --node_num   $node_num  \
  --gamma  0.   \
  --scale_hyperedges 3 \
  --gcn_true True \
  --hgcn_true True \
  --hgat_true False \
  >>./result/PEMS03_hypergnn/LLAMA7b_PEMS03_48_48_hypergnn_naive_order3.txt

