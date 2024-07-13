import pandas as pd


# Read in previously generated data
def retrieve_subframes():
    '''
    Gets partitioned dataframes from source
    :return: list of dataframes
    '''
    subframes = []
    for i in range(0, 8):
        subframes.append(pd.read_csv('../data_inspection/generated_and_cleaned/it_' + str(i) + '.csv'))
    return subframes


def drop_unnamed(subframes):
    for df in subframes:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
    return subframes


def create_all_sentence_set(merged_df):
    df = pd.DataFrame()
    df['text'] = pd.concat([merged_df['question'], merged_df['answers']], ignore_index=True)

    # Add 'is_question' column
    df['is_question'] = [1] * len(merged_df['question']) + [0] * len(merged_df['answers'])
    return df


from nltk import word_tokenize
## Stemmer
# Tokenization and stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


from sklearn.model_selection import train_test_split


def get_train_test(df):
    # reset index, create train + Test splits
    df.reset_index(drop=True, inplace=True)
    return train_test_split(df['text'], df['is_question'], test_size=0.2, random_state=7)


# Split in half
def halve_df(a_df):
    midpoint = len(a_df) // 2
    subframes = [a_df.iloc[i * midpoint: (i + 1) * midpoint] for i in range(2)]
    return subframes


def offest_df(incorrect):
    # Get last value
    final_val = incorrect['answers'].iloc[-1]
    print(final_val)

    # Shift values in the 'y' column down by one row
    incorrect['answers'] = incorrect['answers'].shift(1)

    # Replace the last value of 'y' at position 0
    incorrect.iloc[0, incorrect.columns.get_loc('answers')] = final_val
    return incorrect


def create_corpus(correct, incorrect):
    corpus = pd.DataFrame()
    corpus['text'] = pd.concat([correct['question'], correct['answers'], incorrect['question'], incorrect['answers']],
                               ignore_index=True)
    corpus.reset_index(drop=True, inplace=True)
    return corpus


def find_reference_index(str_val, corpus):
    for index, row in corpus.iterrows():
        if row['text'] == str_val:
            return index
    return -1

def calculate_cosign_similarity(index_a, index_b, sim_matrix):
    return sim_matrix[index_a, index_b]
