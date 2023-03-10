import os
import re
import numpy as np
import torch
import tqdm
import datasets
from utils import arg_parser, CBleu
from model.gpt2_dataset import GPT2Dataset
from model.scored_dataset import ScoredDataset
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, GPT2ForSequenceClassification,RobertaTokenizer, T5ForConditionalGeneration, CodeGenForCausalLM
print(torch.version.cuda)
print(torch.cuda.is_available())
class Computer:
    def __init__(self, args, model, tokenizer, dataset, bleu_tokenizer=None, chunk=-1):
        self.model = model
        self.tokenizer = tokenizer
        self.bleu_tokenizer = bleu_tokenizer
        self.white_space_id = [220,197]
        self.args = args
        self.dataset = dataset
        self.useful_columns = {
            'gpt2': ['input_ids','answers'],
            'gpt2': ['input_ids','answers'],
            'estimator': ['input_ids','gpt2_bleu'],
            'estimator_bleu': ['input_ids','gpt2_bleu'],
            't5': ['input_ids', 'answers'],
            't5_bleu': ['input_ids', 'answers'],
            'codegen': ['input_ids', 'answers']
        }
        self.bleu = datasets.load_metric('./cached/bleu/bleu.py')
        self.chunk = chunk

        try:
            if model == 't5':
                cbleu_corpus = []
                for row in self.dataset.dataset['train']['input_ids']:
                    code = self.tokenizer.decode(row, skip_special_tokens=True)
                    code_ids = self.bleu_tokenizer(code)['input_ids']
                    code_ids = [str(i) for i in code_ids]
                    cbleu_corpus.append(code_ids)
                self.cbleu = CBleu(cbleu_corpus)
            else:
                code_ids = []
                self.cbleu = CBleu([[str(i) for i in d] for d in self.dataset.dataset['train']['input_ids']])
        except KeyError:
            self.cbleu = None

    def compute(self, name):
        args = self.args
        my_dataset = self.dataset.dataset[name]
        if self.chunk != -1: 
            my_dataset = self.dataset.dataset[name].shard(num_shards=5, index=self.chunk)
        train_dataset = my_dataset.with_format(type='torch',columns=self.useful_columns[args.model])
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
         
        if args.model in ['gpt2','codegen']:
            score_dict = self.gen_score_for_gpt2(data_loader)
        elif args.model.startswith('estimator'):
            score_dict = self.gen_score_for_estimator(data_loader)
        elif args.model == 't5':
            score_dict = self.gen_score_for_t5(data_loader)
        else:
            raise Exception('unknown mode')

        for k,v in score_dict.items():
            my_dataset = my_dataset.add_column(k, v)
        save_dir = f'{args.language}_{args.model}_{args.metric}'
        if args.is_dev:
            save_dir += '_dev'
        save_path = os.path.join(args.data_path,save_dir,name if self.chunk == -1 else f'{name}_{self.chunk}')
        my_dataset.save_to_disk(save_path)
    
    def gen_score_for_gpt2(self, data_loader):
        args = self.args
        bleu_scores = []
        cbleu_scores = []
        completed_codes = []

        for batch in tqdm.tqdm(data_loader):
            answers = batch['answers'].numpy()[0].tolist()
            input_ids = batch['input_ids'].to(args.device)
            outputs = self.model.generate(input_ids, max_new_tokens=args.min_query_len, pad_token_id=self.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True, min_length=0)
            sequences = outputs['sequences'].cpu().numpy()
            output_seq = sequences[0][len(input_ids[0]):].tolist()
            if self.tokenizer.eos_token_id in output_seq:
                output_seq = output_seq[:output_seq.index(self.tokenizer.eos_token_id)]
            if len(output_seq) == 0:
                bleu_scores.append({'bleu':0.0})
                cbleu_scores.append(0.0)
                completed_codes.append('<empty generated>')
                continue
            completed_str = self.tokenizer.decode(output_seq)
            if len(answers) == 0:
                bleu_scores.append({'bleu':0.0})
                cbleu_scores.append(0.0)
                completed_codes.append(completed_str)
                continue
            bleu_score = self.bleu.compute(predictions=[[str(i) for i in output_seq]], references=[[[str(i) for i in answers]]],smooth=True)
            bleu_scores.append(bleu_score)
            cbleu_scores.append(self.cbleu.sentence_bleu([str(i) for i in answers], [str(i) for i in output_seq]))
            completed_codes.append(completed_str)

        bleu_scores = [i['bleu'] for i in bleu_scores]
        print(bleu_scores[:500])
        print(np.mean(bleu_scores))
        return {
            f'{args.model}_bleu': bleu_scores,
            f'{args.model}_cbleu': cbleu_scores,
            f'{args.model}_completed': completed_codes
        }
    
    def gen_score_for_estimator(self, data_loader):
        est_scores = []
        
        for batch in tqdm.tqdm(data_loader):
            input_ids = batch['input_ids'].to(args.device)
            logits = self.model(input_ids).logits
            score = logits[0][0].item()
            est_scores.append(score)

        return {
            'est_score': est_scores,
        }
    
    def gen_score_for_t5(self, data_loader):
        bleu_scores = []
        cbleu_scores = []
        completed_codes = []
        extra_id = 32098

        for batch in tqdm.tqdm(data_loader):
            answers = batch['answers'].numpy()[0].tolist()[1:-1]
            answers = answers[:args.min_query_len]
            input_ids = batch['input_ids'].to(args.device)
            outputs = self.model.generate(input_ids, max_new_tokens=3 + args.min_query_len, pad_token_id=self.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True, min_length=0)
            output_seq = outputs['sequences'].cpu().numpy()[0].tolist()
            if extra_id in output_seq:
                output_seq = output_seq[:output_seq.index(extra_id)]
          
            completed_code = self.tokenizer.decode(output_seq, skip_special_tokens=True)

           

            if len(completed_code) == 0:
                bleu_scores.append({'bleu':0.0})
                cbleu_scores.append(0.0)
                completed_codes.append('<empty generated>')
                continue

            completed_code_ids = self.bleu_tokenizer(completed_code)['input_ids']
            if len(answers) == 0:
                bleu_scores.append({'bleu':0.0})
                cbleu_scores.append(0.0)
                completed_codes.append(completed_code)
                continue
            answer_code = self.tokenizer.decode(answers, skip_special_tokens=True)
            answer_code_ids = self.bleu_tokenizer(answer_code)['input_ids']
            if len(answer_code_ids) == 0:
                bleu_scores.append({'bleu':0.0})
                cbleu_scores.append(0.0)
                completed_codes.append(completed_code)
                continue

            if len(completed_code_ids) == 0:
                bleu_scores.append({'bleu':0.0})
                cbleu_scores.append(0.0)
                completed_codes.append('<empty generated>')
                continue
            bleu_score = self.bleu.compute(predictions=[[str(i) for i in completed_code_ids]], references=[[[str(i) for i in answer_code_ids]]],smooth=True)
            bleu_scores.append(bleu_score)
            cbleu_scores.append(self.cbleu.sentence_bleu([str(i) for i in answer_code_ids], [str(i) for i in completed_code_ids]))
            completed_codes.append(completed_code)

        bleu_scores = [i['bleu'] for i in bleu_scores]
        return {
            f'{args.model}_bleu': bleu_scores,
            f'{args.model}_cbleu': cbleu_scores,
            f'{args.model}_completed': completed_codes
        }

if __name__ == '__main__':
    args = arg_parser()
    bleu_tokenizer = None
    if args.model == 't5':
        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        bleu_tokenizer = GPT2TokenizerFast.from_pretrained('./cached/gpt2/')
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
        tokenizer.pad_token = tokenizer.eos_token

    if args.model == 'gpt2':
        dataset = GPT2Dataset(os.path.join(args.data_path, args.dataset_name), tokenizer, is_dev=args.is_dev, mode=args.mode, max_pos_length=args.text_length, language=args.language)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path).to(args.device)
        raise ValueError
    elif args.model.startswith('estimator'):
        dataset = ScoredDataset(os.path.join(args.data_path, args.dataset_name), tokenizer, is_dev=args.is_dev, mode=args.mode,metric=args.metric)
        print(len(dataset.dataset['test']))
        model = GPT2ForSequenceClassification.from_pretrained(args.checkpoint_path).to(args.device)
    elif args.model == 'gpt2':
        dataset = ScoredDataset(os.path.join(args.data_path, args.dataset_name), tokenizer, is_dev=args.is_dev, mode=args.mode)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path).to(args.device)
    elif args.model == 't5':
        dataset = ScoredDataset(os.path.join(args.data_path, args.dataset_name), tokenizer, is_dev=args.is_dev, mode=args.mode, model=args.model, language=args.language)
        model = T5ForConditionalGeneration.from_pretrained(args.cache_path).to(args.device)
    elif args.model == 'codegen':
        dataset = ScoredDataset(os.path.join(args.data_path, args.dataset_name), tokenizer, is_dev=args.is_dev, mode=args.mode, model=args.model, language=args.language)
        # model = CodeGenForCausalLM.from_pretrained(args.cache_path, device_map="auto", load_in_8bit=True)
        model = CodeGenForCausalLM.from_pretrained(args.cache_path).to(args.device)
    model.eval()
    computer = Computer(args, model, tokenizer, dataset, bleu_tokenizer=bleu_tokenizer, chunk=args.chunk)
    if args.model in ['gpt2','t5','codegen']:
        computer.compute('train')
    computer.compute('test')
    