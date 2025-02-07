## ST-SepNet: Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs

Welcome to ST-SepNet's GitHub repository! This project aims to explore the application of hypergraph spatio-temporal learning and Large Language Models (LLMs) in traffic flow demand forecasting. By combining these two advanced techniques, we have developed a novel prediction framework capable of capturing the complex dynamics of transportation networks more precisely and enabling efficient processing of large-scale data.

<p align="center">
<img src="img/model.png" height = "600" alt="" align=center />
</p>





### Data 
The data can be obtained and downloaded from ([Google Drive](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)), and makedir path ```dataset/``` and put dataset in ```dataset/```.




### Large Language Models

The pretrained models can be downloaded from the links in the Table as below,  and makedir path ```huggingface/``` and put pretrained models in ```huggingface/```. For example, ```huggingface/BERT```

| Model                  | Parameters | LLM Dimension |
|------------------------|------------|---------------|
| [BERT](https://huggingface.co/google-bert/bert-base-uncased)                   | 110M       | 768           |
| [GPT-2](https://huggingface.co/openai-community/gpt2)               | 124M       | 768           |
| [GPT-3](https://huggingface.co/TurkuNLP/gpt3-finnish-large)                | 7580M      | 4096          |
| [LLAMA-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)               | 1230M      | 2048          |
| [LLAMA-7B](https://huggingface.co/huggyllama/llama-7b)               | 6740M      | 4096          |
| [LLAMA-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)               | 8000M      | 4096          |
| [DeepSeek-Qwen1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)      | 1500M      | 1536          |


### Our research baselines models refer to the following works and their repository code.


STG4Traffic: {A} Survey and Benchmark of Spatial-Temporal Graph Neural Networks for Traffic Prediction. [[Paper]](https://arxiv.org/abs/2307.00495)[[Code]](https://github.com/trainingl/STG4Traffic?utm_source=catalyzex.com).
```tex
@article{DBLP:journals/corr/abs-2307-00495,
  author       = {Xunlian Luo and Chunjiang Zhu and Detian Zhang and Qing Li},
  title        = {STG4Traffic: {A} Survey and Benchmark of Spatial-Temporal Graph Neural
                  Networks for Traffic Prediction},
  journal      = {CoRR},
  volume       = {abs/2307.00495},
  year         = {2023}
}
```

Deep Time Series Models: A Comprehensive Survey and Benchmark. [[Paper]](https://arxiv.org/abs/2407.13278)[[Code]](https://github.com/thuml/Time-Series-Library).
```tex
@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}

```
