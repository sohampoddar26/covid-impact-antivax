#Tweet preprocessing and data preparation

import pandas as pd
import preprocessor as p

label_dict = {'unnecessary': 'The tweet indicates vaccines are unnecessary, or that alternate cures are better.',
              'mandatory': 'Against mandatory vaccination — The tweet suggests that vaccines should not be made mandatory.',
              'pharma': 'Against Big Pharma — The tweet indicates that the Big Pharmaceutical companies are just trying to earn money, or the tweet is against such companies in general because of their history.',
              'conspiracy': 'Deeper Conspiracy — The tweet suggests some deeper conspiracy, and not just that the Big Pharma want to make money (e.g., vaccines are being used to track people, COVID is a hoax).',
              'political': 'Political side of vaccines — The tweet expresses concerns that the governments / politicians are pushing their own agenda though the vaccines.',
              'country': 'Country of origin — The tweet is against some vaccine because of the country where it was developed / manufactured.',
              'rushed': 'Untested / Rushed Process — The tweet expresses concerns that the vaccines have not been tested properly or that the published data is not accurate.',
              'ingredients': 'Vaccine Ingredients / technology — The tweet expresses concerns about the ingredients present in the vaccines (eg. fetal cells, chemicals) or the technology used (e.gmRNA vaccines can change your DNA)',
              'side-effect': 'Side Effects / Deaths — The tweet expresses concerns about the side effects of the vaccines, including deaths caused.',
              'ineffective': 'Vaccine is ineffective — The tweet expresses concerns that the vaccines are not effective enough and are useless.',
              'religious': 'Religious Reasons — The tweet is against vaccines because of religious reasons',
              'none': 'No specific reason stated in the tweet, or some reason other than the given ones.'}

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')


def preprocess(text):
    filtered_text = p.clean(text)
    return filtered_text


df_train["tweet_filtered"] = df_train["tweet"].apply(preprocess)
df_test["tweet_filtered"] = df_test["tweet"].apply(preprocess)

inputs_list = []
targets_list = []
for i, row in df_train.iterrows():
    tweet = row['tweet_filtered']
    labels = row['labels'].split(' ')
    prompt = "Instruction: First read the task description. " \
             "There could be multiple categories description for a tweet.\n\n " \
             "Task: Multi-label Text Classification \n\n Description: Generate label description for  the given texts. \n\n" + tweet

    targets = []
    for label in labels:
        target = label_dict[label]
        targets.append(target)
    targets = ''.join(targets)
    inputs_list.append(prompt)
    targets_list.append(targets)

df_train['input'] = inputs_list
df_train['target'] = targets_list

inputs_list = []
targets_list = []
for i, row in df_val.iterrows():
    tweet = row['tweet_filtered']
    labels = row['labels'].split(' ')
    prompt = "Instruction: First read the task description. " \
             "There could be multiple categories description for a tweet.\n\n " \
             "Task: Multi-label Text Classification \n\n Description: Generate label description for  the given " \
             "texts. \n\n" + tweet

    targets = []
    for label in labels:
        target = label_dict[label]
        targets.append(target)
    targets = ''.join(targets)
    inputs_list.append(prompt)
    targets_list.append(targets)

df_test['input'] = inputs_list
df_test['target'] = targets_list

df_train.to_csv('./train_caves.csv')
df_test.to_csv('./test_caves.csv')
