from collections import Counter, defaultdict
import json
import os
import pickle
import shutil

import pandas as pd
from sdv.datasets.demo import download_demo, get_available_demos
from sdv.metadata.multi_table import MultiTableMetadata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sdv._utils import train_foreign_key_detector

def dump_relationships(metadata, outdir):
    relationships = set()
    for relation in metadata.relationships:
        relationships.add((
            relation['parent_table_name'],
            relation['parent_primary_key'],
            relation['child_table_name'],
            relation['child_foreign_key']
        ))
    with open(f'{outdir}/relationships.pkl', 'wb') as f:
        pickle.dump(relationships, f)

def store_datasets():
    if os.path.exists('test_set'):
        answer = input('Test set already exists. Press "y" to overwrite: ')
        if answer != 'y':
            return
        shutil.rmtree('test_set')

    os.mkdir('test_set')
    for demo_name in get_available_demos('multi_table')['dataset_name']:
        outdir = f'test_set/{demo_name}'
        os.mkdir(outdir)
        data, metadata = download_demo('multi_table', demo_name)
        for table_name, table_data in data.items():
            table_data.to_csv(f'{outdir}/{table_name}.csv', index=False)

        metadata.save_to_json(f'{outdir}/metadata.json')
        dump_relationships(metadata, outdir)

def confusion_matrix(set1, set2):
    true_positive, false_positive, false_negative = set(), set(), set()
    for key in set1:
        if key in set2:
            true_positive.add(key)
        else:
            false_positive.add(key)
    
    for key in set2:
        if key not in set1:
            false_negative.add(key)
    
    return {
        'True Positive': true_positive,
        'False Positive': false_positive,
        'False Negative': false_negative
    }

def accuracy(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def evaluate():
    total, i, tp, fp, fn = 0, 0, 0, 0, 0
    # total
    with open('evaluation.txt', 'w') as file:
        demo_names = get_available_demos('multi_table')['dataset_name']
        #demo_names = ['world_v1']
        for demo_name in demo_names:
            with open(f'test_set/{demo_name}/relationships.pkl', 'rb') as f:
                true_relationships = pickle.load(f)
            with open(f'predicted/{demo_name}/relationships.pkl', 'rb') as f:
                predicted_relationships = pickle.load(f)

            cm = confusion_matrix(predicted_relationships, true_relationships)
            ac = accuracy(true_relationships, predicted_relationships)
            file.write(f'{demo_name}\n')
            file.write(f'Confusion Matrix: {cm}\n')
            file.write(f'Num Foreign Keys: {len(cm['True Positive']) + len(cm['False Positive']) + len(cm['False Negative'])}\n')
            file.write(f'Num True Positive: {len(true_relationships)}\n')
            file.write(f'Num False Positive: {len(cm["False Positive"])}\n')
            file.write(f'Num False Negative: {len(cm["False Negative"])}\n')
            file.write(f'Accuracy: {ac}\n\n')
            total += ac
            i += 1
            tp += len(cm["True Positive"])
            fp += len(cm["False Positive"])
            fn += len(cm["False Negative"])

        file.write(f'Average Accuracy: {total / i}') # It's actually the Jaccard index
        file.write(f'\nNum True Positive: {tp}')
        file.write(f'\nNum False Positive: {fp}')
        file.write(f'\nNum False Negative: {fn}')

def predict():
    if os.path.exists('predicted'):
        #answer = input('Predicted relationships already exist. Press "y" to overwrite: ')
        #if answer != 'y':
        #    return
        shutil.rmtree('predicted')

    os.mkdir('predicted')
    for demo_name in os.listdir('test_set'):
        os.mkdir(f'predicted/{demo_name}')
        data = {}
        for table_name in os.listdir(f'test_set/{demo_name}'):
            if table_name.endswith('.csv'):
                data[table_name[:-4]] = pd.read_csv(f'test_set/{demo_name}/{table_name}', low_memory=False)

        metadata = MultiTableMetadata()
        metadata = metadata.load_from_json(f'test_set/{demo_name}/metadata.json')
        metadata.relationships = []
        metadata._detect_relationships_hard_coded(data)
        dump_relationships(metadata, f'predicted/{demo_name}')

def visualize_metadata(dataset):
    with open(f'test_set/{dataset}/metadata.json', 'r') as f:
        metadata = json.load(f)
    metadata = MultiTableMetadata.load_from_dict(metadata)
    fig = metadata.visualize()
    fig.view()

def add_metadata():
    metadata = MultiTableMetadata()
    metadata.detect_from_csvs('instacart')
    metadata.save_to_json(f'instacart/metadata.json')

#store_datasets()
#predict()
#evaluate()
#visualize_metadata('world_v1')
#train_foreign_key_detector()
add_metadata()