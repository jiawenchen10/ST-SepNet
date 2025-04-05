# Test baseline model
for model_name in FEDformer ETSformer TimesNet iTransformer Informer DLinear Autoformer Mamba PatchTST TimeMixer  

do
  # BIKE-inflow datasets
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /public/chenjiawen/Time-Series-Library-main/dataset/BIKE/ \
    --data_path inflow.csv \
    --model_id BIKE_Inflow_48_48 \
    --model $model_name\
    --data bike_inflow\
    --features M \
    --seq_len 48 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 295 \
    --dec_in 295 \
    --c_out 295 \
    --des 'Exp' \
    --itr 1 \
    >> result//${model_name}_BIKE_Inflow.txt



  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /public/chenjiawen/Time-Series-Library-main/dataset/BIKE/ \
    --data_path outflow.csv \
    --model_id BIKE_Outflow_48_48 \
    --model $model_name\
    --data bike_outflow\
    --features M \
    --seq_len 48 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 295 \
    --dec_in 295 \
    --c_out 295 \
    --des 'Exp' \
    --itr 1 \
    >> result//${model_name}_BIKE_Outflow.txt


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /public/chenjiawen/Time-Series-Library-main/dataset/BJ/ \
    --data_path BJ.csv \
    --model_id BIKE_BJ_48_48 \
    --model $model_name\
    --data BJ\
    --features M \
    --seq_len 48 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 500 \
    --dec_in 500 \
    --c_out 500 \
    --des 'Exp' \
    --itr 1 \
    >> result//${model_name}_BJ.txt



  # METR datasets
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /public/chenjiawen/Time-Series-Library-main/dataset/METR/ \
    --data_path METR.csv \
    --model_id METR_48_48 \
    --model $model_name\
    --data METR \
    --features M \
    --seq_len 48 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 207 \
    --dec_in 207 \
    --c_out 207 \
    --des 'Exp' \
    --itr 1 \
    >> result//${model_name}_METR.txt



  # PEMS03 datsetss
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /public/chenjiawen/Time-Series-Library-main/dataset/PEMS/ \
    --data_path PEMS03.csv \
    --model_id PEMS03_48_48 \
    --model $model_name \
    --data PEMS03 \
    --features M \
    --seq_len 48 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --itr 1 \
    >> result/${model_name}_PEMS03.txt
done