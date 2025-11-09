#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import os

files = ['all_support_data.csv', 'all_perturbed_data_with_judge_v1.csv', 'all_neutral_data.csv']
output_file = 'all_data_with_judge_with_fig_ref_v1.csv'
df = pd.DataFrame()
for file in files:
    temp_df = pd.read_csv(file)
    print(file.split('.csv')[0], '- Unique classes', temp_df['class'].value_counts(), '\n\n')
    df = pd.concat([df, temp_df], ignore_index=True)
print('Total records:', df.shape, '\n\n')
print('unique classes and their counts:', df['class'].value_counts(), '\n\n')

df.to_csv(output_file, index=False)

print('Combined data is saved to', output_file)
print('End.')