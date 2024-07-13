# Read in previously generated data-> Tested
def retrieve_subframes():
    '''
    Gets partitioned dataframes from source
    :return: list of dataframes
    '''
    subframes = []
    for i in range(0, 8):
        subframes.append(pd.read_csv('../data_inspection/generated_and_cleaned/it_' + str(i) + '.csv'))
    return subframes

# drop unnamed columns from list of dataframes-> sentences
def drop_unnamed(subframes):
    for df in subframes:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
    return subframes

# Concatenate question and answer series vertically
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


import pandas as pd

# Data Acquisition
# Read in previously generated data
subframes = retrieve_subframes()
for df in subframes:
    print(len(df))


# Test retrieve_subframes
subframes = retrieve_subframes()

assert len(subframes) == 8
for frame in subframes:
    assert len(frame) == 500

assert set(subframes[0].columns) == set(['Unnamed: 0', 'question', 'answers'])

# test remove_unnamed
subframes = drop_unnamed(subframes)
for frame in subframes:
    assert set(subframes[0].columns) == set(['question', 'answers'])

# test create_all_sentence_set
test_dict = {'question' : [" a question"],
             'answers': ['an answer']}

test_frame = pd.DataFrame(test_dict)
converted = create_all_sentence_set(test_frame)
assert len(converted) == 2
assert set(converted.columns) == set(['text', 'is_question'])


# test tokenize and stem
test_string = "Unit tests . ^ are very imporTant"
assert set(['unit', 'test', '.', '^', 'are', 'veri', 'import']) == set(tokenize_and_stem(test_string))

# test get_train_test
test_dict_2 = {'text' : [" a question", "question2", " a question", "question2", " a question", "question2", " a question", "question2", " a question", "question2"],
             'is_question': ['1', '2', '1', '2', '1', '2', '1', '2', '1', '2']}
test_frame_2 = pd.DataFrame(test_dict_2)
test_splits = get_train_test(test_frame_2)
assert len(test_splits[0]) == 8
assert len(test_splits[3]) == 2

# Test halve_df
halved = halve_df(test_frame_2)
for frame in halved:
    assert len(frame) == 5
    assert set(frame.columns) == set(test_frame_2.columns)


# Test Offset 1
test_dict_3 = {'question' : [" a question", "question2", " a question", "question2", " a question", "question2", " a question", "question2", " a question", "question2"],
             'answers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
test_frame_3 = pd.DataFrame(test_dict_3)
test_frame_4 = offest_df(test_frame_3.copy())
assert test_frame_3['answers'].sum() == test_frame_4['answers'].sum()
assert test_frame_3.loc[1, 'answers'] != test_frame_4.loc[1, 'answers']
assert test_frame_3.loc[1, 'answers'] == test_frame_4.loc[2, 'answers']