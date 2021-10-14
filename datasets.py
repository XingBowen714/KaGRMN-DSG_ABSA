from torch.nn.utils.rnn import pad_sequence
import argparse
import codecs
import json
import linecache
import logging
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from copy import copy, deepcopy

import nltk
import numpy as np
import simplejson as json
import torch
#from allennlp.modules.elmo import batch_to_ids
#from lxml import etree
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def load_datasets_and_vocabs(args):
    train,train_graph, test, test_graph = get_dataset(args.dataset_name)

    # Our model takes unrolled data, currently we don't consider the MAMS cases(future experiments)
    train_samples= get_arranged_data(train,train_graph, args)
    test_samples = get_arranged_data(test, test_graph, args)

    logger.info('****** After unrolling ******')
    logger.info('Train set size: %s', len(train_samples))
    logger.info('Test set size: %s,', len(test_samples))

    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(
        train_samples+test_samples, args)
    if args.embedding_type == 'glove':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.glove_embedding = embedding

    train_dataset = ASBA_Depparsed_Dataset(
        train_samples, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    test_dataset = ASBA_Depparsed_Dataset(
        test_samples, args, word_vocab, dep_tag_vocab, pos_tag_vocab)

    return train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab


def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

def read_sparse_depgraph(file_path):
    f = open(file_path, 'rb')
    idx2graph = pickle.load(f)
    f.close()
    return idx2graph

def get_dataset(dataset_name):
    
    rest_train = 'data/semeval14/restaurant_train'
    rest_test = 'data/semeval14/restaurant_test'

    laptop_train = 'data/semeval14/laptop_train'
    laptop_test = 'data/semeval14/laptop_test'

    res15_train = 'data/semeval15/train'
    res15_test = 'data/semeval15/test'

    ds_train = {'rest': rest_train,
                'laptop': laptop_train, 'res15': res15_train}
    ds_test = {'rest': rest_test,
               'laptop': laptop_test, 'res15': res15_test}

    train = list(read_sentence_depparsed(ds_train[dataset_name] + '_biaffine_depparsed_des.json'))
    train_graph = read_sparse_depgraph(ds_train[dataset_name] + '.graph')
    logger.info('# Read %s Train set: %d', dataset_name, len(train))

    test = list(read_sentence_depparsed(ds_test[dataset_name] + '_biaffine_depparsed_des.json'))
    test_graph = read_sparse_depgraph(ds_test[dataset_name] + '.graph')
    logger.info("# Read %s Test set: %d", dataset_name, len(test))
    return train, train_graph, test, test_graph


def reshape_dense_dependency_tree(as_start, as_end, dependencies, multi_hop=False, add_non_connect=False, tokens=None, max_hop = 5):

    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(2)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'

    return dep_tag, dep_idx, dep_dir

def reshape_sparse_dep_graph(as_start, as_end, graph):
    
    if as_end == as_start + 1:
        return graph
    d_idx = []
    length = graph.shape[0]
    for i in range(as_start+1, as_end):
        graph[:, as_start] = graph[:, as_start] + graph[:, i]
        graph[as_start, :] = graph[as_start, :] + graph[i, :]
        d_idx.append(i)
    
    graph = np.delete(graph, d_idx, axis = 0)
    graph = np.delete(graph, d_idx, axis = 1)
    graph[graph>1] = 1
    length2 = graph.shape[0]
    assert length  == length2 - 1  + as_end - as_start, ' here it is' + '{0}, {1}, {2}'.format(d_idx, as_start, as_end)
    return graph
    
def get_arranged_data(input_data, graph, args):

    opinionated_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
                        'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    processed_samples = []
    

    # Make sure the tree is successfully built.
    zero_dep_counter = 0

    # Sentiment counters
    total_counter = defaultdict(int)
    mixed_counter = defaultdict(int)

    logger.info('*** Start sparse dependency graph and dense dependency tree reshaping) ***')
    tree_samples = []
    idx = 0
    # for seeking 'but' examples
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]


        pos_class = e['tags']

        # Iterate through aspects in a sentence and reshape the dependency tree.
        if args.noak:
            description = e['aspect']
        else:
            description = e['asp_des']
        description = word_tokenize(description.lower())
        aspect = word_tokenize(e['aspect'])
        frm = e['asp_position']
        to = e['asp_position'] + e['aspect_len']

        #print('aspect:{0}'.format(aspect))
        # Center on the aspect.
        dep_tag, dep_idx, dep_dir = reshape_dense_dependency_tree(frm, to, e['dependencies'],
                                                       multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)
        #try:
        assert graph[idx].shape[0] == len(e['tokens'])
        sparse_graph = reshape_sparse_dep_graph(frm, to, graph[idx])

        if len(dep_tag) == 0:
            zero_dep_counter += 1
            as_sent = e['aspect_sentiment'][i][0].split()
            as_start = e['tokens'].index(as_sent[0])
            # print(e['tokens'], e['aspect_sentiment'], e['dependencies'],as_sent[0])
            as_end = e['tokens'].index(
                as_sent[-1]) if len(as_sent) > 1 else as_start + 1
            print("Debugging: as_start as_end ", as_start, as_end)
            dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(as_start, as_end, e['dependencies'],
                                                       multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)
            if len(dep_tag) == 0:  # for debugging
                print("Debugging: zero_dep",
                      e['aspect_sentiment'][i][0], e['tokens'])
                print("Debugging: ". e['dependencies'])
            else:
                zero_dep_counter -= 1
        idx = idx + 1
        if len(e['tokens']) < to:
            print(e['tokens'], aspect, frm, to)
            assert 1==0, 'context too short'
        processed_samples.append(
            {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'aspect_len': e['aspect_len'], 'asp_des': description,'sentiment': e['polarity'],
                'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
             'aspect_start': frm + 1, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir':dep_dir,'dependencies': e['dependencies'], 'sparse_graph': sparse_graph})

    logger.info('Dependency tree reshaping done!\n')

    return processed_samples


def load_and_cache_vocabs(data, args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # Build or load word vocab and glove embeddings.
    # Elmo and bert have it's own vocab and embeddings.
    if args.embedding_type == 'glove':
        cached_word_vocab_file = os.path.join(
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',
                        args.dataset_name)
            word_vocab = build_text_vocab(data)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(
            args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.glove_dir)
            word_vecs = load_glove_embedding(
                word_vocab['itos'], args.glove_dir, 0.25, args.embedding_dim)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        word_vocab = None
        word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab


def load_glove_embedding(word_list, glove_dir, uniform_scale, dimension_size):
    glove_words = []
    with open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r') as fopen:
        for line in fopen:
            glove_words.append(line.strip().split(' ')[0])
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(os.path.join(
                glove_dir, 'glove.840B.300d.txt'), word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(
                line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(
                np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors


def _default_unk_index():
    return 1


def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['sentence']

        counter.update(s)

    itos = ['[PAD]', '[UNK]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_pos_tag_vocab(data, vocab_size=2000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for d in data:
        tags = d['tags']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


class ASBA_Depparsed_Dataset(Dataset):
    '''
    Convert examples to features, numericalize text to ids.
    '''

    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items = e['dep_tag_ids'], \
            e['pos_class'], e['text_len'], e['aspect_len'], e['des_len'], e['sentiment'],\
            e['dep_rel_ids'], e['predicted_heads'], e['aspect_position'], e['dep_dir_ids'], \
            e['aspect_start'], e['sparse_graph']
        if self.args.embedding_type == 'glove':
            non_bert_items = e['sentence_ids'], e['aspect_ids']
            items_tensor = non_bert_items + items
            items_tensor = tuple(torch.tensor(t) for t in items_tensor)
        elif self.args.embedding_type == 'elmo':
            items_tensor = e['sentence_ids'], e['aspect_ids']
            items_tensor += tuple(torch.tensor(t) for t in items)
        else:  # bert
            if self.args.pure_bert:
                bert_items = e['input_cat_ids'], e['segment_ids']
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
            else:
                bert_items = e['input_ids'], e['word_indexer'], e['w_idx'], e['input_aspect_ids'], \
                e['aspect_indexer'], e['input_cat_ids'], e['segment_ids'], e['input_des_ids'], \
                e['des_indexer']
                # segment_id
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
        return items_tensor

    def convert_features_bert(self, i):
        """
        BERT features.
        convert sentence to feature. 
        """
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        # tokenizer = self.args.tokenizer
        aspect_start = self.data[i]['aspect_start']
        aspect_len = self.data[i]['aspect_len']
        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []
        des_tokens = []
        des_indexer = []

        # review
        for word in self.data[i]['sentence']:
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            tokens.extend(word_tokens)

            word_indexer.append(token_idx)

        # aspect
        for word in self.data[i]['aspect']:
            word_aspect_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(aspect_tokens)
            aspect_tokens.extend(word_aspect_tokens)
            aspect_indexer.append(token_idx)

        # description
        #print(self.data[i]['asp_des'])
        for word in self.data[i]['asp_des']:
            word_des_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(des_tokens)
            des_tokens.extend(word_des_tokens)
            
            des_indexer.append(token_idx)

        w_idx = word_indexer[:aspect_start] + word_indexer[aspect_start+aspect_len-1:]
        

        tokens = [cls_token] + tokens + [sep_token]
        aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
        des_tokens = [cls_token] + des_tokens + [sep_token]

        word_indexer = [i+1 for i in word_indexer]
        aspect_indexer = [i+1 for i in aspect_indexer]
        des_indexer = [i+1 for i in des_indexer]
        #print(des_indexer)

        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(aspect_tokens)
        input_des_ids = self.args.tokenizer.convert_tokens_to_ids(des_tokens)

        # check len of word_indexer equals to len of sentence.
        assert len(word_indexer) == len(self.data[i]['sentence'])
        assert len(aspect_indexer) == len(self.data[i]['aspect'])
        assert len(des_indexer) == len(self.data[i]['asp_des'])



        if self.args.pure_bert:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
        else:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
            self.data[i]['input_ids'] = input_ids
            self.data[i]['word_indexer'] = word_indexer
            self.data[i]['input_aspect_ids'] = input_aspect_ids
            self.data[i]['aspect_indexer'] = aspect_indexer
            self.data[i]['input_des_ids'] = input_des_ids
            self.data[i]['des_indexer'] = des_indexer
            self.data[i]['w_idx'] = w_idx

    def convert_features(self):

        for i in range(len(self.data)):
            aspect_start = self.data[i]['aspect_start']
            aspect_len = self.data[i]['aspect_len']
            if self.args.embedding_type == 'glove':
                self.data[i]['sentence_ids'] = [self.word_vocab['stoi'][w]
                                                for w in self.data[i]['sentence']]
                self.data[i]['aspect_ids'] = [self.word_vocab['stoi'][w]
                                              for w in self.data[i]['aspect']]
            elif self.args.embedding_type == 'elmo':
                self.data[i]['sentence_ids'] = self.data[i]['sentence']
                self.data[i]['aspect_ids'] = self.data[i]['aspect']
            else:  # self.args.embedding_type == 'bert'
                self.convert_features_bert(i)

            self.data[i]['text_len'] = len(self.data[i]['sentence'])
            self.data[i]['des_len'] = len(self.data[i]['asp_des'])
            self.data[i]['aspect_position'] = [0] * self.data[i]['text_len']
            try:  # find the index of aspect in sentence
                for j in range(self.data[i]['aspect_start'], self.data[i]['to']):
                    self.data[i]['aspect_position'][j] = 1
            except Exception as e:
                #print(e)
                for term in self.data[i]['aspect']:
                #    print(121,self.data[i]['aspect'])
                    self.data[i]['aspect_position'][self.data[i]['sentence'].index(term)] = 1
                #assert 1==0

            self.data[i]['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in self.data[i]['dep_tag']]
            self.data[i]['dep_tag_ids'] = self.data[i]['dep_tag_ids'][:aspect_start] + \
            self.data[i]['dep_tag_ids'][aspect_start+aspect_len - 1:]
            assert len(self.data[i]['dep_tag_ids']) == self.data[i]['text_len'] - self.data[i]['aspect_len'] + 1, '{0}, {1}'.format(self.data[i]['text_len'], self.data[i]['aspect_len'])
            assert self.data[i]['sparse_graph'].shape[0] == self.data[i]['text_len'] - self.data[i]['aspect_len'] + 1, '{0}, {1}, {2}'.format(self.data[i]['text_len'], self.data[i]['aspect_len'],\
             self.data[i]['sparse_graph'].shape[0])
            assert len(self.data[i]['dep_tag_ids']) == self.data[i]['sparse_graph'].shape[0], '{0}, {1}'.format(len(self.data[i]['dep_tag_ids']), self.data[i]['sparse_graph'].shape)
            self.data[i]['dep_dir_ids'] = [idx
                                           for idx in self.data[i]['dep_dir']]
            self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                             for w in self.data[i]['tags']]
            self.data[i]['aspect_len'] = len(self.data[i]['aspect'])

            
            #self.data[i]['aspect_start'] = self.data[i]['from']

            self.data[i]['dep_rel_ids'] = [self.dep_tag_vocab['stoi'][r]
                                           for r in self.data[i]['predicted_dependencies']]



def my_collate_bert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    '''
    input_ids, word_indexer, w_idx, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, \
    input_des_ids, des_indexer, dep_tag_ids, pos_class, text_len, aspect_len, des_len, \
    sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids, aspect_start, sparse_graph= zip(*batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    des_len = torch.tensor(des_len)
    aspect_start = torch.tensor(aspect_start)
    sentiment = torch.tensor(sentiment)
    max_len = max([len(p) for p in dep_tag_ids]) 
    #print(max_len) 
    #print(text_len.size()[0])
    sparse_graph_list = []
    for graph in sparse_graph:
        #print(graph)
        graph = graph.numpy()
        #print(graph)
        #print(graph.shape)
        graph = np.pad(graph, ((0,max_len-graph.shape[0]),(0,max_len-graph.shape[0])), 'constant')
        sparse_graph_list.append(graph)
    sparse_graph = torch.tensor(sparse_graph_list)
    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_aspect_ids = pad_sequence(input_aspect_ids, batch_first=True, padding_value=0)
    input_cat_ids = pad_sequence(input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value =0)
    input_des_ids = pad_sequence(input_des_ids, batch_first=True, padding_value =0)
    # indexer are padded with 1, for ...
    word_indexer = pad_sequence(word_indexer, batch_first=True, padding_value=1)
    aspect_indexer = pad_sequence(aspect_indexer, batch_first=True, padding_value=1)
    des_indexer = pad_sequence(des_indexer, batch_first=True, padding_value=1)
    w_idx = pad_sequence(w_idx, batch_first=True, padding_value=1)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)

    input_ids = input_ids[sorted_idx]
    input_aspect_ids = input_aspect_ids[sorted_idx]
    input_des_ids = input_des_ids[sorted_idx]

    word_indexer = word_indexer[sorted_idx]
    aspect_indexer = aspect_indexer[sorted_idx]
    des_indexer = des_indexer[sorted_idx]

    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    des_len = des_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    dep_heads = dep_heads[sorted_idx]
    aspect_start = aspect_start[sorted_idx]
    w_idx = w_idx[sorted_idx]
    sparse_graph = sparse_graph[sorted_idx]

    return input_ids, word_indexer, w_idx, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, \
    input_des_ids, des_indexer, dep_tag_ids, pos_class, text_len, aspect_len, des_len, sentiment, dep_rel_ids, \
    dep_heads, aspect_positions, dep_dir_ids, aspect_start, sparse_graph
def my_collate_bert_eval(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    '''
    input_ids, word_indexer, w_idx, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, \
    input_des_ids, des_indexer, dep_tag_ids, pos_class, text_len, aspect_len, des_len, \
    sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids, aspect_start, sparse_graph= zip(*batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    des_len = torch.tensor(des_len)
    aspect_start = torch.tensor(aspect_start)
    sentiment = torch.tensor(sentiment)
    max_len = max([len(p) for p in dep_tag_ids]) 
    #print(max_len) 
    #print(text_len.size()[0])
    sparse_graph_list = []
    for graph in sparse_graph:
        #print(graph)
        graph = graph.numpy()
        #print(graph)
        #print(graph.shape)
        graph = np.pad(graph, ((0,max_len-graph.shape[0]),(0,max_len-graph.shape[0])), 'constant')
        sparse_graph_list.append(graph)
    sparse_graph = torch.tensor(sparse_graph_list)
    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_aspect_ids = pad_sequence(input_aspect_ids, batch_first=True, padding_value=0)
    input_cat_ids = pad_sequence(input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value =0)
    input_des_ids = pad_sequence(input_des_ids, batch_first=True, padding_value =0)
    # indexer are padded with 1, for ...
    word_indexer = pad_sequence(word_indexer, batch_first=True, padding_value=1)
    aspect_indexer = pad_sequence(aspect_indexer, batch_first=True, padding_value=1)
    des_indexer = pad_sequence(des_indexer, batch_first=True, padding_value=1)
    w_idx = pad_sequence(w_idx, batch_first=True, padding_value=1)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)


    return input_ids, word_indexer, w_idx, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, \
    input_des_ids, des_indexer, dep_tag_ids, pos_class, text_len, aspect_len, des_len, sentiment, dep_rel_ids, \
    dep_heads, aspect_positions, dep_dir_ids, aspect_start, sparse_graph
