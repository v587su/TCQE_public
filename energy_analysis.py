import torch
from deepspeed.profiling.flops_profiler import get_model_profile, FlopsProfiler
from transformers import  GPT2TokenizerFast, GPT2ForSequenceClassification,AutoModelForCausalLM,RobertaTokenizer, RobertaModel,AutoTokenizer, AutoModelForSeq2SeqLM, CodeGenForCausalLM


def input_constructor(batch_size, seq_len, tokenizer, name):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    if name in ['gpt2','codet5','gpt-neo']:
        # copy input_ids
        labels = inputs['input_ids'].clone()
    elif name == 'dce':
        labels = torch.tensor([1] * batch_size)


    inputs = dict(inputs)
    if name in ['gpt2','dce','gpt-neo','gpt-j']:
        inputs.update({"labels": labels})
    elif name in ['codet5']:
        inputs.update({'decoder_input_ids': labels})
    
    for k, v in inputs.items():
        inputs[k] = v.to('cuda:0')
    return inputs




targets = [{
    'path':'./results/python_python_codegen_estimator_bleu_with_space/best',
    'model': GPT2ForSequenceClassification,
    'tokenizer':GPT2TokenizerFast.from_pretrained('./cached/gpt2'), 
    'name': 'dce'
    },{
    # 'path':'./results/gpt2-highlr/checkpoint-356860',
    'path':'./cached/gpt2',
    'model': AutoModelForCausalLM,
    'tokenizer':GPT2TokenizerFast.from_pretrained('./cached/gpt2'),
    'name': 'gpt2'
    },{
    'path':'./cached/codegen-350M-multi',
    'model': CodeGenForCausalLM,
    'tokenizer': GPT2TokenizerFast.from_pretrained('./cached/codegen-350M-multi'),
    'name': 'codegen'
    }]


for t in targets:
    tokenizer = t['tokenizer']
    tokenizer.pad_token = tokenizer.eos_token
    model = t['model'].from_pretrained(t['path']).to('cuda:0')
    batch_size = 1
    seq_len = 256
    name = t['name']
    input_shape = (batch_size, seq_len)
    # tracker = OfflineEmissionsTracker(country_iso_code='CHN',cloud_provider='gcp',cloud_region='asia-east2')
    # tracker.start()
    prof = FlopsProfiler(model)
    model.eval()
    inputs = input_constructor(batch_size, seq_len, tokenizer, name)
   

    for _ in range(1):
        if name == 'dce':
            _ = model(inputs['input_ids'])
        else:
            _ = model.generate(inputs['input_ids'], max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, eos_token_id=12345)
    exit = False
    for max_tk_num in [10,20,50]:
        print(name,max_tk_num)
        prof.start_profile(ignore_list=None)
        if name == 'dce':
            _ = model(inputs['input_ids'])
            exit = True
        else:
            _ = model.generate(inputs['input_ids'], max_new_tokens=max_tk_num, pad_token_id=tokenizer.eos_token_id, eos_token_id=12345)

        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
        params = prof.get_total_params()
        prof.print_model_profile(profile_step=1,
                                module_depth=-1,
                                top_modules=1,
                                detailed=False,
                                output_file=None)
        prof.end_profile()
        if exit:
            break


    