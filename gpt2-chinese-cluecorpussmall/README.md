---
language: Chinese
datasets: CLUECorpusSmall
widget: 
- text: "这是很久之前的事情了"


---


# Chinese GPT2 Model

## Model description

The model is used to generate Chinese texts. You can download the model either from the [GPT2-Chinese Github page](https://github.com/Morizeyao/GPT2-Chinese), or via HuggingFace from the link [gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall).

## How to use

You can use the model directly with a pipeline for text generation:

```python
>>> from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
>>> tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
>>> model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
>>> text_generator = TextGenerationPipeline(model, tokenizer)   
>>> text_generator("这是很久之前的事情了", max_length=100, do_sample=True)
    [{'generated_text': '这是很久之前的事情了 ， 我 曾 经 把 这 个 当 做 一 种 思 想 的 传 承 ， 或 者 是 人 生 的 回 顾 ， 当 时 我 们 是 一 个 刚 刚 加 入 的 时 候 就 想 要 加 入 他 们 ， 于 是 我 们 每 天 看 到 他 们 ， 加 上 他 们 的 各 种 不 可 思 议 的 行 为 ， 直 到 现 在 ， 我 们 的 人 生 才 完 整 起 来 。'}]
```

## Training data

[CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020/) is used as training data. 

## Training procedure

The model is pre-trained by [UER-py](https://github.com/dbiir/UER-py/) on [Tencent Cloud](https://cloud.tencent.com/). We pre-train 1,000,000 steps with a sequence length of 128 and then pre-train 250,000 additional steps with a sequence length of 1024. 

Stage1:

```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path cluecorpussmall_lm_seq128_dataset.pt \
                      --seq_length 128 --processes_num 32 --data_processor lm 
```

```
python3 pretrain.py --dataset_path cluecorpussmall_lm_seq128_dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --config_path models/gpt2/config.json \
                    --output_model_path models/cluecorpussmall_gpt2_seq128_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 1000000 --save_checkpoint_steps 100000 --report_steps 50000 \
                    --learning_rate 1e-4 --batch_size 64
```

Stage2:

```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path cluecorpussmall_lm_seq1024_dataset.pt \
                      --seq_length 1024 --processes_num 32 --data_processor lm 
```

```
python3 pretrain.py --dataset_path cluecorpussmall_lm_seq1024_dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/cluecorpussmall_gpt2_seq128_model.bin-1000000 \
                    --config_path models/gpt2/config.json \
                    --output_model_path models/cluecorpussmall_gpt2_seq1024_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 250000 --save_checkpoint_steps 50000 --report_steps 10000 \
                    --learning_rate 5e-5 --batch_size 16
```

Finally, we convert the pre-trained model into Huggingface's format:

```
python3 scripts/convert_gpt2_from_uer_to_huggingface.py --input_model_path cluecorpussmall_gpt2_seq1024_model.bin-250000 \
                                                        --output_model_path pytorch_model.bin \
                                                        --layers_num 12
```

### BibTeX entry and citation info

```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}

@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}
```