import spacy_pkuseg as pkuseg
import pandas as pd
import ast
import jieba
from tqdm import tqdm
from collections import defaultdict
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def pkuseg_process(input_tags):
    output_tokens = set()
    output_column = []
    seg = pkuseg.pkuseg(postag=False)

    for tags in tqdm(input_tags,desc="pkuseg_processing"):
        output_single = set()
        for tag in tags:
            tokens = seg.cut(tag)
            # output_single.update(set(token[0] for token in tokens))
            output_single.update(set(tokens))
        output_column.append(output_single)    
        output_tokens.update(output_single)  
    print(len(output_tokens))
    return output_tokens, output_column

def ckiptagger_process(input_tags):
    
    def make_word_pos_sentence(word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            print(f"{word}({pos})", end="\u3000")
        print()
        return
    
    output_tokens = set()
    output_column = []
    
    ws = WS("./model/data")
    pos = POS("./model/data")
    debug_times = 0
    for tags in tqdm(input_tags, desc="ckiptagger_processing"):
        # debug_times += 1
        # if debug_times == 3:
        #     exit()
        output_single = set()
        output_tags = []
        list_tags = list(tags)
        word_sentence_list = ws(list_tags)
        # word_sentence_list = ws(list(tags), sentence_segmentation=True, segment_delimiter_set = {"，",",", "。", ":", "?","？", "!","！", ";", "；"})
        # pos_sentence_list = pos(word_sentence_list)
        # for tag, tag_pos in zip(word_sentence_list, pos_sentence_list):
        #     output_tags.extend(zip(tag, tag_pos))
        for tag in word_sentence_list:
            output_tags.extend(tag)
        output_single.update(set(output_tags))
        output_column.append(output_single)    
        output_tokens.update(output_single)
    
    del ws
    del pos
    
    print(len(output_tokens))
    return output_tokens, output_column

def jieba_process(input_tags):
    output_tokens = set()
    output_column = []
    seg = pkuseg.pkuseg(postag=True)

    for tags in tqdm(input_tags,desc="jieba_processing"):
        output_frequncy = {}
        output_single = set()
        for tag in tags:
            tokens = list(jieba.cut(tag))
            output_single.update(set(tokens))
            for token in tokens:
                if token not in output_frequncy:
                    output_frequncy[token] = 0
                output_frequncy[token] += 1
            # output_single.update(set(token[0] for token in tokens))
        output_column.append(output_frequncy)    
        output_tokens.update(output_single) 
    print(len(output_tokens))
    return output_tokens, output_column


def pipeline(input, selected_method):
    input_tags= [ast.literal_eval(tags) for tags in input['Tags']]
    
    process_functions = {
        'pkuseg': pkuseg_process,
        'ckiptagger': ckiptagger_process,
        'jieba': jieba_process
    }
    process_func = process_functions.get(selected_method)
    
    if not process_func:
        raise ValueError(f"Unknown method selected: {selected_method}")

    output_tokens, output_column = process_func(input_tags)
    output_file = f'./output/{selected_method}_output_all.csv' #Book
    column_file = f'./output/{selected_method}_output_frequency_column.csv' #Book
    # output_file = f'./output_movie/{selected_method}_output_all.csv' #Movie
    # column_file = f'./output_movie/{selected_method}_output_frequency_column.csv' #Movie
    # dataframe_all = pd.DataFrame(list(output_tokens), columns=['token', 'word_class'])
    dataframe_all = pd.DataFrame(list(output_tokens), columns=['token'])
    dataframe_column = pd.DataFrame(zip(input['Book'], output_column), columns=['Book', 'Tokens_Frequency']) #Book
    # dataframe_column = pd.DataFrame(zip(input['Movie'], output_column), columns=['Movie', 'Tokens_Frequency']) #Movie
    dataframe_all.to_csv(output_file, index=False)
    dataframe_column.to_csv(column_file, index=False)

    print(f"Files saved: {output_file}, {column_file}")
     

if __name__ == "__main__":
    input = pd.read_csv('./data/selected_book_top_1200_data_tag.csv') #Book
    # input = pd.read_csv('./data/selected_movie_top_1200_data_tag.csv') #Movie
    # methods = ['pkuseg','ckiptagger','jieba']
    methods = ['jieba']
    for method in methods:
        pipeline(input, method)

    
