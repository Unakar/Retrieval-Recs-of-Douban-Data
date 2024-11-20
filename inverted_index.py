# 构建倒排表 并用跳表存储
import ast
import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import pickle
import re

book_num = 1200

class Node:
    def __init__(self, key=None, value=None, next=None, down=None):
        self.key = key
        self.value = value
        self.next = next
        self.down = down

class SkipList:
    def __init__(self, num):
        self.levels = 0
        self.num = num
        self.head = Node(value=num,key=None,next=None,down=None)
        # print("Success")
    
    def get_all_keys(self, sort=True):
        cur = self.head
        while cur.down:
            cur = cur.down
        
        key_value_pairs = []
        cur = cur.next
        while cur:
            key_value_pairs.append((cur.key, cur.value))
            cur = cur.next
        if sort:
            key_value_pairs.sort(key=lambda x: x[1], reverse=True)  # 按文档数量升序排序
            return key_value_pairs
        else:
            return key_value_pairs

    def compress(self):
        cur_head = self.head
        while cur_head:
            cur = cur_head
            last_id = 0
            while cur.next:
                cur = cur.next
                tempt = cur.key
                cur.key = cur.key - last_id
                last_id = tempt
            cur_head = cur_head.down
        

    def search(self, key):
        cur = self.head
        while cur:
            while cur.next and cur.next.key < key:
                cur = cur.next
            if cur.next and cur.next.key == key:
                return cur.next.value
            cur = cur.down
        return None

    def insert(self, key, value):
        stack = []
        cur = self.head
        while cur:
            while cur.next and cur.next.key < key:
                cur = cur.next
            if cur.next and cur.next.key == key:
                cur.next.value += value
                cur = cur.next
                while cur.down:
                    cur.down.value += value
                    cur = cur.down
                return
            stack.append(cur)
            cur = cur.down

        insert_new = True
        down_node = None
        self.num += 1
        while insert_new and stack:
            cur = stack.pop()
            new_node = Node(key=key, value=value, next=cur.next, down=down_node)
            cur.next = new_node
            down_node = new_node
            insert_new = random.randint(0, 1)

        if insert_new:
            first_node = Node(key=key, value=value, down=down_node)
            new_head = Node(key=None, value=self.num,next=first_node,down=self.head)
            self.head = new_head
            self.levels += 1

def get_similar_words_mapping(file_path):
    data = pd.read_csv(file_path)
    words = data['word']
    similar_words = data['similar_words']
    similar_words_mapping = {}
    for i in tqdm(range(len(words)), desc='similar_words_mapping'):
        word = words[i]
        if word is None or (isinstance(word, float) and pd.isnull(word)):
            continue
        swords = similar_words[i]
        if swords is None or (isinstance(swords, float) and pd.isnull(swords)):
            continue
        swords = ast.literal_eval(swords)
        for sword in swords:
            similar_words_mapping[sword] = word        
    return similar_words_mapping
    

def build_inverted_index_from_all_tokens(file_path):
    inverted_index = defaultdict(list)
    data = pd.read_csv(file_path)
    tokens = data['word']
    for token in tokens:
        if token is None or (isinstance(token, float) and pd.isnull(token)):
            continue
        inverted_index[token] = {}
    return inverted_index

def construct_inverted_index_from_frequency(all_tokens_path, file_frequency_path, similar_words_mapping):
    inverted_index = build_inverted_index_from_all_tokens(all_tokens_path)
    data = pd.read_csv(file_frequency_path)
    for i in tqdm(range(len(data)), desc='inverted_index_inserting'):
        book = data['Book'][i]
        tokens_frequency = data['Tokens_Frequency'][i]
        tokens_frequency = ast.literal_eval(tokens_frequency)
        for token, frequency in tokens_frequency.items():
            if token not in similar_words_mapping:
                continue
            token = similar_words_mapping[token]
            if token is None or (isinstance(token, float) and pd.isnull(token)):
                continue
            if token in inverted_index:
                if book in inverted_index[token]:
                    inverted_index[token][book] += frequency
                else:
                    inverted_index[token][book] = frequency
    return inverted_index

def convert_inverted_index_to_skip_list(inverted_index):
    inverted_index_skip_list = {}
    for word, index in tqdm(inverted_index.items(), desc='converting_to_skiplist'):
        # print("word: ", word, "index: ", len(index))
        skip_list = SkipList(0)
        for id, frequency in index.items():
            skip_list.insert(id, frequency)
        inverted_index_skip_list[word] = skip_list
    return inverted_index_skip_list

def compress_inverted_index_skip_list(method, inverted_index_skip_list):
    compressed_inverted_index_skip_list_file_path = 'output/' + method + '_compressed_inverted_index_skip_list.pkl'
    compressed_inverted_index_skip_list = {}
    for word, skiplist in inverted_index_skip_list.items():
        compress_skiplist = skiplist.compress()
        compressed_inverted_index_skip_list[word] = compress_skiplist
    save_inverted_index_skip_list_to_file(compressed_inverted_index_skip_list, compressed_inverted_index_skip_list_file_path)
    return compressed_inverted_index_skip_list

def save_inverted_index_skip_list_to_file(inverted_index_skip_list, file_path):
    pickle.dump(inverted_index_skip_list, open(file_path, 'wb'))
    
def load_inverted_index_skip_list_from_file(file_path):
    return pickle.load(open(file_path, 'rb'))

def query_word(similar_words_mapping, inverted_index_skip_list, word):
    if word not in similar_words_mapping:
        print(f"'{word}' not found")
    word = similar_words_mapping[word]
    result = inverted_index_skip_list[word].get_all_keys()
    if result:
        print(f"Documents containing '{word}': {result}")
    else:
        print(f"'{word}' not found")

def pipeline(method):
    all_tokens_path = 'output/' + method + '_res_sim_words_all.csv'
    file_frequency_path = 'output/' + method + '_output_frequency_column.csv'
    inverted_index_skip_list_file_path = 'output/' + method + '_inverted_index_skip_list.pkl'
    compressed_inverted_index_skip_list_file_path = 'output/' + method + '_compressed_inverted_index_skip_list.pkl'
    similar_words_mapping = get_similar_words_mapping(all_tokens_path)
    inverted_index = construct_inverted_index_from_frequency(all_tokens_path, file_frequency_path, similar_words_mapping)
    # save inverted index to file
    inverted_index_df = pd.DataFrame(inverted_index.items(), columns=['word', 'inverted_index'])
    inverted_index_df.to_csv('output/' + method + '_inverted_index.csv', index=False)
    inverted_index_skip_list = convert_inverted_index_to_skip_list(inverted_index)
    save_inverted_index_skip_list_to_file(inverted_index_skip_list, inverted_index_skip_list_file_path)
    # query_word(similar_words_mapping, inverted_index_skip_list, '渐兄')
    return inverted_index_skip_list

def parse_query(query, similar_words_mapping, inverted_index_skip_list):
    query = query.replace(" ", "")
    print(query)
    precedence = {"not": 3, "and": 2, "or": 1}
    operators = []
    operands = []
    index_length = {}

    pattern = r'(and|or|not|\(|\))'
    query = re.split(pattern, query)
    query = [token for token in query if token]
    # print(query)

    def apply_operator():
        operator = operators.pop()
        if operator == "not":
            operand = operands.pop()
            operand_length = index_length[operand] if operand in index_length else 0
            if operand.startswith("(") and operand.endswith(")"):
                query_string = f"not{operand}"
            else:
                query_string = f"not({operand})"
            operands.append(query_string)
            index_length[query_string] = 1200 - operand_length
        else:
            right = operands.pop()
            left = operands.pop()
            right_length = index_length[right] if right in index_length else 0
            left_length = index_length[left] if left in index_length else 0
            if operator == "and":
                if left_length > right_length:
                    query_string = f"({right} {operator} {left})"
                else:
                    query_string = f"({left} {operator} {right})"
                index_length[query_string] = min(left_length, right_length)
            else:
                if left_length < right_length:
                    query_string = f"({right} {operator} {left})"
                else:
                    query_string = f"({left} {operator} {right})"
                index_length[query_string] = left_length + right_length
            operands.append(query_string)

    for character in query:
        if character == "(":
            operators.append(character)
        elif character == ")":
            while operators and operators[-1] != "(":
                apply_operator()
            operators.pop()  # 弹出 "("
        elif character in precedence:
            while (operators and operators[-1] != "(" and
                   precedence[operators[-1]] >= precedence[character]):
                apply_operator()
            operators.append(character)
        else:
            character = similar_words_mapping[character] if character in similar_words_mapping else character
            operands.append(character)
            index_length[character] = inverted_index_skip_list[character].num if character in inverted_index_skip_list else 0

    while operators:
        apply_operator()
    return operands[0]

def not_skiplist(skiplist):
    # 从 selected_book_top_1200_data_tag.csv 中获取所有的书籍id
    book_reader = pd.read_csv('./data/selected_book_top_1200_data_tag.csv')
    book_all = book_reader['Book']
    book_to_delete = [id for id,_ in skiplist.get_all_keys()]
    book_delete_not = list(set(book_all) - set(book_to_delete))
    new_skiplist = SkipList(0)
    for id in book_delete_not:
        new_skiplist.insert(id, 1)
    return new_skiplist
    
def and_skiplists(left_skiplist, right_skiplist):
    dict_a = dict(left_skiplist.get_all_keys(sort=False))
    dict_b = dict(right_skiplist.get_all_keys(sort=False))
    intersection_keys = dict_a.keys() & dict_b.keys()
    
    intersection = {key: dict_a[key] + dict_b[key] for key in intersection_keys}
    new_skiplist = SkipList(0)
    for id, frequency in intersection.items():
        new_skiplist.insert(id, frequency)
    return new_skiplist

def or_skiplists(left_skiplist, right_skiplist):
    dict_right = right_skiplist.get_all_keys(sort=False)
    for key, value in dict_right:
        left_skiplist.insert(key, value)
    return left_skiplist
    

class ExpressionNode:
    def __init__(self, isnot, Skiplist, value=None):
        self.isnot = isnot
        self.skiplist = Skiplist

def process_query(query, inverted_index_skip_list):
    query = query.replace(" ", "")
    # print(query)
    precedence = {"not": 3, "and": 2, "or": 1}
    operators = []
    operands = []
    Expression_SkipLists = {}

    pattern = r'(and|or|not|\(|\))'
    query = re.split(pattern, query)
    query = [token for token in query if token]
    # print(query)

    def apply_operator():
        operator = operators.pop()
        if operator == "not":
            operand = operands.pop()
            if operand.startswith("(") and operand.endswith(")"):
                query_string = f"not{operand}"
            else:
                query_string = f"not({operand})"
            operands.append(query_string)
            Expression_SkipLists[query_string] = not_skiplist(Expression_SkipLists[operand])
        else:
            right = operands.pop()
            left = operands.pop()
            right_skiplist = Expression_SkipLists[right]
            left_skiplist = Expression_SkipLists[left]
            if operator == "and":
                new_skiplist = and_skiplists(left_skiplist, right_skiplist)
            else:
                new_skiplist = or_skiplists(left_skiplist, right_skiplist)
            query_string = f"({left} {operator} {right})"
            operands.append(query_string)
            Expression_SkipLists[query_string] = new_skiplist

    for character in query:
        if character == "(":
            operators.append(character)
        elif character == ")":
            while operators and operators[-1] != "(":
                apply_operator()
            operators.pop()  # 弹出 "("
        elif character in precedence:
            while (operators and operators[-1] != "(" and
                   precedence[operators[-1]] >= precedence[character]):
                apply_operator()
            operators.append(character)
        else:
            character = similar_words_mapping[character] if character in similar_words_mapping else character
            operands.append(character)
            if character in inverted_index_skip_list:
                Expression_SkipLists[character] = inverted_index_skip_list[character]
            else:
                Expression_SkipLists[character] = SkipList(0)
            
    while operators:
        apply_operator()
    return Expression_SkipLists[operands[0]]

class FrontEncodingNode:
    def __init__(self, prefix_len, suffix, skip_list):
        self.prefix_len = prefix_len
        self.suffix = suffix
        self.skip_list = skip_list
    
def front_encoding(inverted_index_skip_list, method):
    word_list = list(inverted_index_skip_list.keys())
    word_list.sort()
    prev = ''
    FrontEncodingNodes = []
    for word in word_list:
        prefix_len = 0
        while prefix_len < len(prev) and prefix_len < len(word) and prev[prefix_len] == word[prefix_len]:
            prefix_len += 1
        FrontEncodingNodes.append(FrontEncodingNode(prefix_len, word[prefix_len:], inverted_index_skip_list[word]))
        prev = word
    file_path = 'output/'+method+'_front_encoding.pkl'
    pickle.dump(FrontEncodingNodes, open(file_path, 'wb'))
    return FrontEncodingNodes

def front_encoding_search(FrontEncodingNodes, word):
    prev = ''
    for node in FrontEncodingNodes:
        current_word = prev[:node.prefix_len] + node.suffix
        if current_word == word:
            return node.skip_list
        prev = current_word
    return None

class BlockStorageNode:
    def __init__(self, k, dictionary_string, block_pointers, block_lengths, skip_list_all):
        self.k = k
        self.dictionary_string = dictionary_string
        self.block_pointers = block_pointers
        self.block_lengths = block_lengths
        self.skip_list_all = skip_list_all

def front_encoding_search(FrontEncodingNodes, word):
    prev = ''
    for node in FrontEncodingNodes:
        current_word = prev[:node.prefix_len] + node.suffix
        if current_word == word:
            return node.skip_list
        prev = current_word
    return None

def block_encoding(k, inverted_index_skip_list, method):
    word_list = list(inverted_index_skip_list.keys())
    word_list.sort()
    dictionary_string = ''.join(word_list)
    block_pointers = []
    block_lengths = []
    start_pos = 0
    for i in range(0, len(word_list), k):
        block_pointers.append(start_pos)
        lengths = []
        for word in word_list[i:i+k]:
            lengths.append(len(word))
            start_pos += len(word)
        block_lengths.append(lengths)
    skip_list_all = []
    for word in word_list:
        skip_list_all.append(inverted_index_skip_list[word])
    CompressBlockStorageNode = BlockStorageNode(k, dictionary_string, block_pointers, block_lengths, skip_list_all)

    file_path = 'output/'+method+'_block_encoding.pkl'
    pickle.dump(CompressBlockStorageNode, open(file_path, 'wb'))
    return CompressBlockStorageNode

def block_encoding_search(CompressBlockStorageNode, word):
    k = CompressBlockStorageNode.k
    dictionary_string = CompressBlockStorageNode.dictionary_string
    block_pointers = CompressBlockStorageNode.block_pointers
    block_lengths = CompressBlockStorageNode.block_lengths
    skip_list_all = CompressBlockStorageNode.skip_list_all
    length = 0
    for i in range(len(block_pointers)):
        block_pointer = block_pointers[i] #start
        for j in range(k):
            if length >= len(dictionary_string):
                return None
            current_word = dictionary_string[block_pointer:block_pointer+block_lengths[i][j]]
            if current_word == word:
                return skip_list_all[k*i+j]
            block_pointer += block_lengths[i][j]
            length += block_lengths[i][j]
    return None


if __name__ == '__main__':
    methods = ['jieba']
    for method in methods:
        inverted_index_skip_list = pipeline(method)
        all_tokens_path = 'output/' + method + '_res_sim_words_all.csv'
        file_frequency_path = 'output/' + method + '_output_frequency_column.csv'
        inverted_index_skip_list_file_path = 'output/' + method + '_inverted_index_skip_list.pkl'
        similar_words_mapping = get_similar_words_mapping(all_tokens_path)
        # inverted_index_skip_list = load_inverted_index_skip_list_from_file(inverted_index_skip_list_file_path)
        query = parse_query("动作and(剧情 or 科幻) and not (恐怖 or开心) and 友情 or (游戏 and not 爱情)",similar_words_mapping,inverted_index_skip_list)
        result_skiplist = process_query(query, inverted_index_skip_list)
        print(result_skiplist.get_all_keys())
        
        # compressed_inverted_index_skip_list = compress_inverted_index_skip_list(method, inverted_index_skip_list)
        FrontEncodingNodes = front_encoding(inverted_index_skip_list, method)
        result_skiplist_fe = front_encoding_search(FrontEncodingNodes, '川端康成')
        print(result_skiplist_fe.get_all_keys())
        
        print(inverted_index_skip_list['川端康成'].get_all_keys())
        
        print("Block Encoding")
        CompressBlockStorageNode = block_encoding(4, inverted_index_skip_list, method)
        result_skiplist_be = block_encoding_search(CompressBlockStorageNode, '川端康成')   
        print(result_skiplist_be.get_all_keys())     

# 法兰克福 欧美
# 川端康成 日本 小说