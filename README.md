# Source code of the paper "Don't Complete It! Preventing Unhelpful Code Completion for Productive and Sustainable Neural Code Completion Systems"

### Datasets
Download CSN and COFIC into the `datasets` folder. The datasets are available at [CSN(python)](https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip) and [COFIC](https://drive.google.com/file/d/1Ai0WMYrIGQQLqBC180mzUVDSbpkgO6uD/view)

### LCMs
#### GPT-2
Download the pre-trained GPT-2 model from [here](https://huggingface.co/gpt2/tree/main) and put it into the `cached` folder.
The model needs to be finetuned on the dataset before using it for code completion. 
The command for finetuning GPT-2 on CSN(Python) is as follows:
```
python train_gpt2.py --batch_size 8 --run_name gpt2 --epoch 10 --text_length 256 --data_path datasets/python/final/jsonl --mode train --language python
```

The command for finetuning GPT-2 on COFIC(Java) is as follows:
```
python train_gpt2.py --batch_size 8 --run_name gpt2 --epoch 10  --text_length 256 --data_path ./datasets/COFIC.jsonl --language java --mode train
```


#### CodeGen
Download the pre-trained CodeGen model from [here](https://huggingface.co/Salesforce/codegen-350M-mono/tree/main) and put it into the `cached` folder.
The model doesn't need further finetuning.

### Run

#### Generate training dataset for TCQE
We first generate a dataset for training TCQE. The dataset is generated by querying the LCMs with the training dataset.
For GPT-2:
```
python generate_score.py --checkpoint_path PATH_OF_FINETUNED_GPT2 --mode score --model gpt2 --batch_size 1 --text_length 256 --min_query_len 10 --dataset_name python/final/jsonl --language python (or java)
```
For CodeGen:

```
python generate_score.py --cache_path PATH_OF_CODEGEN --mode score --model codegen --batch_size 1 --text_length 256 --min_query_len 10 --dataset_name python/final/jsonl --language python (or java)
```

### Train TCQE
```
python train_estimator.py --batch_size 8 --run_name A_NAME_AS_YOU_WANT --epoch 30 --data_path PATH_OF_GENERATED_DATASET --metric gpt2_bleu (or gpt2_cbleu, codegen_bleu, codegen_cbleu) --language python (or java)
```

### Generate estimation
```
python generate_score.py --checkpoint_path PATH_OF_TRAINED_TCQE --dataset_name PATH_OF_GENERATED_DATASET --mode eval --batch_size 1 --metric gpt2_bleu (or gpt2_cbleu, codegen_bleu, codegen_cbleu) --model estimator --language python (or java)
```

### Scripts for processing the results
compute accuracy: accuracy_error.py

compute_code_metrics: compute_code_metrics.py

compute_flops: energy_analysis.py