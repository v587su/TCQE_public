import datasets
import math
import pandas as pd
import numpy as np
import random
from datasets import load_from_disk
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import pearsonr



if __name__ == '__main__':
    names = ['random','est_score','coc']
    languages = ['java', 'python']
    metrics = ['gpt2_bleu','gpt2_cbleu','codegen_cbleu','codegen_bleu']
    proportions = [0.01, 0.05, 0.1, 0.25, 1]
    p_names = ['1%','5%','10%','25%', '100%']
    results = []
    for language in languages:
        for metric in metrics:
            print(metric)
            dataset_with_metrics = load_from_disk(f"datasets/{language}_estimator_bleu_{metric}/test_with_metrics")
            gpt2_bleu = np.array(dataset_with_metrics['gpt2_bleu' if metric.startswith('gpt2') else 'codegen_bleu'])        
            gpt2_cbleu = np.array(dataset_with_metrics['gpt2_cbleu' if metric.startswith('gpt2') else 'codegen_cbleu'])
            actual_score = np.array(dataset_with_metrics[metric])
            for name in names:
                est_score = np.array(dataset_with_metrics[name])
                print(est_score.mean())
                thre_values = [np.percentile(est_score,p*100,interpolation='nearest') for p in proportions]
                for i, thre in enumerate(thre_values):
                    thre_mse = ((actual_score[est_score<=thre]-est_score[est_score<=thre])**2).mean()
                    thre_mae = abs(actual_score[est_score<=thre]-est_score[est_score<=thre]).mean()
                 
                    rejected = est_score[est_score<=thre]
                    rejected_real = actual_score[est_score<=thre]
    
                    success_bleu = gpt2_bleu[est_score>thre].mean()
                    fail_bleu = gpt2_bleu[est_score<=thre].mean()
                    success_cbleu = gpt2_cbleu[est_score>thre].mean()
                    fail_cbleu = gpt2_cbleu[est_score<=thre].mean()
                    results.append([language, metric, name, p_names[i], thre, (rejected_real-rejected).mean(), thre_mse, thre_mae,  success_bleu, fail_bleu, success_cbleu, fail_cbleu])
                   
    df = pd.DataFrame(results, columns=['language', 'metric', 'name', 'proportion', 'threshold', 'error', 'mse', 'mae',  'success_bleu', 'fail_bleu', 'success_cbleu', 'fail_cbleu'])

    def pivot_table(df, name):
        tmp_df = df[['language', 'metric', 'name', 'proportion', name]]
        tmp_df['model'] = tmp_df['metric'].apply(lambda x: x.split('_')[0])
        tmp_df['metric'] = tmp_df['metric'].apply(lambda x: x.split('_')[1])
        tmp_df = tmp_df.pivot_table(index=['language', 'model', 'metric', 'name'], columns='proportion', values=name)
        tmp_df.to_csv(f'accuracy_err_{name}.csv')
        return df

    pivot_table(df, 'mae')
    pivot_table(df, 'mse')
    pivot_table(df, 'fail_bleu')
    pivot_table(df, 'fail_cbleu')
    pivot_table(df, 'success_bleu')
    pivot_table(df, 'success_cbleu')
