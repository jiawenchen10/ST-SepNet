from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer,AutoTokenizer, AutoModelForCausalLM, AutoConfig,AutoModel


from layers.Embed import PatchEmbedding
from layers.GraphGCN import GCN
import transformers
from layers.StandardNorm import Normalize
from layers.MTGNN import gtnet as MTGNN
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import pandas as pd
import os
transformers.logging.set_verbosity_error()
import gc

gc.collect()
torch.cuda.empty_cache()


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        
        self.weight_A = nn.Parameter(torch.randn(self.rank, original_layer.in_features))
        self.weight_B = nn.Parameter(torch.randn(original_layer.out_features, self.rank))
        
        nn.init.kaiming_uniform_(self.weight_A, a=sqrt(5))
        nn.init.zeros_(self.weight_B)
        
    def forward(self, x):
        original_out = self.original_layer(x)
        
        # Flatten the input tensor
        batch_size = x.size(0)
        input_dim = x.size(-1)
        x_flat = x.view(-1, input_dim)  # Flatten the input to [batch_size * sequence_length, input_dim]
        
        # Calculate the LoRA output
        lora_out_flat = torch.matmul(self.weight_B, torch.matmul(self.weight_A, x_flat.T)).T
        
        # Adjust the shape back to the original shape
        lora_out = lora_out_flat.view(batch_size, -1, self.weight_B.size(0))
        
        return original_out + lora_out
    



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  
        self.linear = nn.Linear(nf, target_window)  
        self.dropout = nn.Dropout(head_dropout)  

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.adj = torch.tensor(pd.read_csv(os.path.join(configs.root_path, configs.adjacency_path), 
                                            index_col=False,header=None).values).to(device='cuda:0', dtype=torch.float32)
        
        self.gcn = GCN(configs.seq_len, configs.seq_len, configs.seq_len)
        self.mtgnn = MTGNN( gcn_true=True, buildA_true=True,  gcn_depth=2,    
                        num_nodes=configs.node_num, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                        predefined_A= self.adj, static_feat=None,  dropout=0.3,  subgraph_size=20,  
                        node_dim=40,  dilation_exponential=1,  conv_channels=32,   residual_channels=32,  skip_channels=64,   
                        end_channels=128,  seq_length= configs.seq_len,   in_dim=1,   out_dim=configs.seq_len, 
                        layers=3,   propalpha=0.05,  tanhalpha=3,  layer_norm_affline=True)
        self.mtgnn.to(torch.bfloat16)

        if configs.llm_model == 'LLAMA7b':
            self.llama_config = LlamaConfig.from_pretrained('huggingface/llama7b/')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_model = LlamaModel.from_pretrained(
                'huggingface/llama7b/',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
            )

            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggingface/llama7b/',
                trust_remote_code=True,
                local_files_only=True
            )


        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('huggingface/gpt2')
            
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                'huggingface/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'huggingface/gpt2',
                trust_remote_code=True,
                local_files_only=True
            )



        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('huggingface/google_bert')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            self.llm_model = BertModel.from_pretrained(
                'huggingface/google_bert',
                trust_remote_code=True,
                local_files_only=True,
                config=self.bert_config,
            )

            self.tokenizer = BertTokenizer.from_pretrained(
                'huggingface/google_bert',
                trust_remote_code=True,
                local_files_only=True
                )

        else:
            raise Exception('LLM model is not defined')

        def set_module(model, module_path, new_module):
            parts = module_path.split('.')
            parent = model
            for part in parts[:-1]:
                if part.isdigit():  
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        modules_to_replace = []
        for name, module in self.llm_model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, LoRALayer(module, rank=4)))

        for name, new_module in modules_to_replace:
            set_module(self.llm_model, name, new_module)



        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)


        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,adj = None):
        # 融合网络拓扑信息
        x_enc = self.normalize_layers(x_enc, 'norm')
        if len(x_enc.size()) == 4:
            x_enc = torch.squeenze(x_enc)
        B, T, N = x_enc.size()
        
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: " 
                f'The time stamp information states as [month, day, hour, minutes],'
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)




        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))   # (B,N,325,32)
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()  


        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])   #(B,N,F)
        dec_out = dec_out.permute(0, 2, 1).contiguous()  #(B,F,N)
              
        
        dec_out = self.normalize_layers(dec_out, 'denorm')   #(B,F,N)

        

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)



    def forward(self, target_embedding, source_embedding, value_embedding):
        
        
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
    
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        
        out = out.reshape(B, L, -1)


        return self.out_projection(out)
    
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
