import pandas
import fasttext.util
import numpy as np
from tqdm import tqdm
from collections import defaultdict


ft = fasttext.load_model('./model/cc.zh.300.bin/cc.zh.300.bin')
print('Successfully load fasttext model')

def get_vector_similarity(vector1, vector2):
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)
    if norm_1 == 0 or norm_2 == 0:
        return 0
    dot = np.dot(vector1, vector2)
    similarity = dot / (norm_1 * norm_2)
    return similarity

def get_tokenized_words(file_path):
    data = pandas.read_csv(file_path)
    tokenized_words = data['token']
    return tokenized_words

def get_similar_words(tokenized_words):
    similar_words = {}
    to_delete = []
    for i in tqdm(range(len(tokenized_words)), desc='similar_words_processing'):
        word = tokenized_words[i]
        # print('word: '+word)
        if word is None or (isinstance(word, float) and np.isnan(word)):
            continue
        if word in to_delete:
            continue
        j = i + 1
        while j < len(tokenized_words):
            sword = tokenized_words[j]
            if sword is None \
                or (isinstance(sword, float) and np.isnan(sword)):
                j += 1
                continue
            try:
                vector1 = ft.get_word_vector(word)
                vector2 = ft.get_word_vector(sword)
                similarity = get_vector_similarity(vector1, vector2)
                if similarity > 0.65:
                    similar_words[word] = similar_words.get(word, []) + [sword]
                    to_delete.append(sword)              
                    # print(' '+sword+':'+str(similarity))
                j += 1
            except KeyError:
                j += 1
                continue
    for word_to_delete in to_delete:
        tokenized_words = tokenized_words[tokenized_words != word_to_delete]
    return similar_words, tokenized_words

def remove_stop_words(tokenized_words):
    stop_words = []
    with open('./data/stopwords/stop_words.txt', 'r') as f:
        for line in f:
            stop_words.append(line.strip())
    for sw in stop_words:
        tokenized_words = tokenized_words[tokenized_words != sw]
    return tokenized_words

def pipeline(method):
    mapping_words = {}
    input_file_path = './output/' + method + '_output_all.csv'
    output_file_path = './output/' + method + '_res_sim_words_all.csv'
    tokenized_words = get_tokenized_words(input_file_path)
    print('get_tokenized_words: '+str(len(tokenized_words)))
    similar_words, tokenized_words = get_similar_words(tokenized_words)
    print('get_similar_words: '+str(len(tokenized_words)))
    tokenized_words = remove_stop_words(tokenized_words)
    print('remove_stop_words: '+str(len(tokenized_words)))
    for word in tokenized_words:
        maps =  [] if word not in similar_words else similar_words[word]
        maps.append(word)
        mapping_words[word] = maps
        
    # save similar_words to file
    similar_words_df = pandas.DataFrame(mapping_words.items(), columns=['word', 'similar_words'])
    similar_words_df.to_csv(output_file_path, index=False)
    return 

if __name__ == '__main__':
    methods = ['jieba']
    # methods = ['jieba','pkuseg','ckiptagger']
    for method in methods:
        pipeline(method)