import re
import sys
import csv
from typing import Tuple
import torch
import pandas as pd
import torch.utils.data
from collections import Counter
from keybert import KeyBERT
from typing import Dict, Iterable, List, Optional
from collections import Counter, OrderedDict
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

csv.field_size_limit(1000000)

local_model = SentenceTransformer("./local_model")  # Load the saved model


class InputExample(object):
    def __init__(self, guid=None, text=None, user=None, product=None, label=None, category=None, keywordlist=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.user = user
        self.product = product
        self.category = category
        self.keywordlist = keywordlist


class Data(torch.utils.data.Dataset):
    sort_key = None
    def __init__(self, *data):
        assert all(len(data[0]) == len(d) for d in data)
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
       return tuple(d[index] for d in self.data)


class SentenceProcessor(object):
    NAME = 'SENTENCE'

    def get_sentences(self):
        raise NotImplementedError

    def _create_examples(self, documents, type):
        try:
            examples = []
            for (i, line) in tqdm(enumerate(documents), total=len(documents), desc="Creating examples", unit="doc"):
                guid = "%s-%s" % (type, i)
                text = clean_document(line[2])
                kw_model = KeyBERT(local_model)
                keyword_list = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None)
                raw_keyword_list = [item[0] for item in keyword_list]
                line[5] = raw_keyword_list[0:1]
                examples.append(
                    InputExample(guid=guid, user=line[0], product=line[1], text=text, 
                                label=int(line[3]) - 1, category=line[4], keywordlist=line[5])
                )
            return examples
        except Exception as e:
            print(f"Create example error: {e}")

          
    def _read_file(self, dataset):
        # Read the CSV file Add (structure out code)
        pd_reader = pd.read_csv(dataset, header=None, skiprows=0, encoding="utf-8", sep=r'\t\t', engine='python', on_bad_lines='skip')
        documents = []

        # Iterate over the rows
        for i in range(len(pd_reader[0])):
            try:
                # Check if the review (third column) exists and is not empty
                review = pd_reader[3][i]  # Assuming the review is in the third column (index 3)
                document = [pd_reader[0][i], pd_reader[1][i], review, pd_reader[2][i], pd_reader[4][i], []]
                documents.append(document)
            except KeyError:
                # In case the column doesn't exist, skip the row
                
                continue
            except:
                print(type(review), review)
                exit()

        return documents

    def _create_sentences(self, *datasets):
        sentences = []
        for dataset in datasets:
            for id, document in enumerate(dataset):
                #Document construction Add for more columns(structure in code)
                user = document[0]
                product = document[1]
                review = document[2]
                label = int(document[3]) - 1
                category = document[4]
                
                keywordlist = document[5]
                sentences.extend([InputExample(user=user, product=product, text=sentence, label=label, category=category, keywordlist=keywordlist) for
                                  sentence in generate_sents(clean_document(review))])
                # for s in generate_sents(clean_document(review)):
                #     f = open("temp.txt", 'a')
                #     f.write(s+'\n')
        return sentences

    def _creat_sent_doc(self, *datasets):
        import time
        documents = []
        for dataset in datasets:
            for id, document in enumerate(dataset):
                #Document construction: Add for more columns(structure in code)
                user = document[0]
                product = document[1]
                review = document[2]
                label = int(document[3]) - 1
                category = document[4]
                keywordlist = document[5]
                documents.append(InputExample(user=user, product=product, text=generate_sents(clean_document(review)), label=label, category=category, keywordlist=keywordlist))
                print(generate_sents(clean_document(review)))
                print(len(generate_sents(clean_document(review))))
                time.sleep(10)

        return documents


    def _get_attributes(self, *datasets):
      users = Counter()
      products = Counter()
      category = Counter()
      ATTR_MAP = {
          'user': 0,  # assuming indices are integers and not strings
          'product': 1,
          'category': 4
      }
      for dataset in datasets:
          for document in dataset:
              users[document[ATTR_MAP["user"]]] += 1
              products[document[ATTR_MAP["product"]]] += 1
              category[document[ATTR_MAP["category"]]] += 1
      return tuple([users, products, category]) 
    
    def _get_keywords(self, *datasets):
      keyword_counter = Counter()
      
      ATTR_MAP = {
          'keywordlist': 5,
      }
      
      try:
          for dataset in datasets:
              for document in dataset:
                  try:
                      for keyword in document[ATTR_MAP["keywordlist"]]:
                          keyword_counter[keyword] += 1
                  except (IndexError, KeyError, TypeError) as e:
                      print(f"Skipping document due to error: {e}")
                      
      except Exception as e:
          print(f"Unexpected error: {e}")

      return keyword_counter

    def _get_polarized_keywords(self, *datasets, classes):
        poskeywordset = Counter()
        negkeywordset = Counter()
        
        ATTR_MAP = {
            'label': 3,
            'keywordlist': 5,
        }
        
        try:
            for dataset in datasets:
                for document in dataset:
                    try:
                        for keyword in document[ATTR_MAP["keywordlist"]]:
                            if document[ATTR_MAP['label']] > (classes - 1) / 2:
                                poskeywordset[keyword] += 1
                            else:
                                negkeywordset[keyword] += 1
                    except (IndexError, KeyError, TypeError) as e:
                        print(f"Skipping document due to error: {e}")
                        
        except Exception as e:
            print(f"Unexpected error: {e}")

        return poskeywordset, negkeywordset

    # def _get_keywords(self, *datasets):
    #     keywords = set()
        
    #     # userspecificKws = []
    #     ATTR_MAP = {
    #       #'user': 0,  # assuming indices are integers and not strings
    #       'text': 2,
    #     }
    #     for dataset in datasets:
    #         for document in dataset:
    #             # user = document[ATTR_MAP["user"]]
    #             doc = document[ATTR_MAP["text"]]
    #             kw_model = KeyBERT(local_model)
    #             keywordList = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)
    #             rawKeywordList = [item[0] for item in keywordList]
    #             for kw in rawKeywordList:
    #                 keywords.add(kw)
    #                 # userspecificKws.append({kw: user})
    #     return keywords #[keywords,...]
    
    # def _get_keyword_counter(self, *datasets):
    #     keywordCounter = Counter()
    #     # userspecificKws = []
    #     ATTR_MAP = {
    #     # 'user': 0,  # assuming indices are integers and not strings
    #       'text': 2,
    #     }
    #     for dataset in datasets:
    #         for document in dataset:
    #             # user = document[ATTR_MAP["user"]]
    #             doc = document[ATTR_MAP["text"]]
    #             kw_model = KeyBERT(local_model)
    #             keywordList = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)
    #             rawKeywordList = [item[0] for item in keywordList]
    #             for kw in rawKeywordList:
    #                 keywordCounter[kw] += 1
    #                 # userspecificKws.append({kw: user})
    #     #first list is for BERT processing, second is for attribute or mapped embedding processing
    #     return keywordCounter             
        


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


class UnknownUPVecCache(object):
    @classmethod
    def unk(cls, tensor):
        return tensor.uniform_(-0.25, 0.25)


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"sssss", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


# def split_sents(string):
#     string = re.sub(r"[!?]", " ", string)
#     string = re.sub(r"\.{2,}", " ", string)
#     sents = string.strip().split('.')
#     sents = [clean_string(sent) for sent in sents]
#     return filter(lambda x: len(x) > 0, sents)


def clean_document(document):
    if pd.isna(document) or document is None:
          return "" 
    string = re.sub(r"<sssss>", "", document)
    string = re.sub(r" n't", "n't", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'.`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def build_vocab(counter):

    
    try:
        if not counter:
            raise ValueError("Counter is empty; cannot build vocab.")
        
        # Handle non-string and NaN keys without modifying the order or structure
        iterator = ([str(key) if key is not None and key == key else "<unk>" for key in counter.keys()])
        # Pass iterator to build_vocab_from_iterator
        vocab = build_vocab_from_iterator([iterator], specials=["<unk>"])
       
        return vocab
    except Exception as e:
        print(f"Error in build_vocab: {e}")
        print(f"counter: {counter}")
        return None



def generate_sents(docuemnt, max_length=230):
    if isinstance(docuemnt, list):
        docuemnt = docuemnt[0]
    string = re.sub(r"[!?]", " ", docuemnt)
    string = re.sub(r"\.{2,}", " ", string)
    sents = string.strip().split('.')
    sents = [clean_string(sent) for sent in sents]
    n_sents = []
    n_sent = []
    for sent in sents:
        n_sent.extend(sent)
        if len(n_sent) > max_length:
            n_sents.append(" ".join(n_sent))
            n_sent = []
            n_sent.extend(sent)
    n_sents.append(" ".join(n_sent))
    return n_sents

def _truncate_and_pad(tokens, max_length=510, pad_strategy="head"):
    """
    Truncates a sequence in place to the maximum length
    :param tokens:
    :param max_length:
    :param pad_strategy: "head", "tail", "both"
    :return:
    """
    total_length = len(tokens)
    if total_length > max_length:
        if pad_strategy == 'head':
            return ['[CLS]'] + tokens[:max_length] + ['[SEP]']
        if pad_strategy == 'tail':
            return ['[CLS]'] + tokens[-max_length:]+ ['[SEP]']
        if pad_strategy == 'both':
            return ['[CLS]'] + tokens[:128] + tokens[-max_length+128:] + ['[SEP]']
        return
    else:
        return ['[CLS]'] + tokens + ['[SEP]'] + ['<PAD>'] * (max_length-total_length)

class Vocab(nn.Module):
    def __init__(self, tokens):
        super(Vocab, self).__init__()
        self.itos = list(tokens)  # Index-to-string mapping
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}  # String-to-index mapping

    def __getitem__(self, token):
        """Allows `vocab[token]` access like a dictionary."""
        return self.stoi.get(token, self.stoi["<unk>"])  # Return index, default to "<unk>"

    def get_itos(self):
        """Returns the index-to-string mapping."""
        return self.itos

    def get_stoi(self):
        """Returns the index-to-string mapping."""
        return self.stoi

    def __len__(self):
        return len(self.itos)


def build_vocab_from_iterator(iterator: Iterable, min_freq: int = 1, specials: Optional[list] = None, special_first: bool = True):
    """
    Constructs a `Vocab` object from an iterator.

    Args:
        iterator (Iterable): An iterator that yields lists of tokens (e.g., sentences or documents).
        min_freq (int): Minimum frequency for a token to be included in the vocabulary.
        specials (list): List of special tokens (e.g., ["<unk>", "<pad>"]).
        special_first (bool): Whether to place special tokens at the start of the vocabulary.

    Returns:
        Vocab: A vocabulary object.
    """
    counter = Counter()
    
    # Count token frequencies
    for tokens in iterator:
        counter.update(tokens)
    
    # Add special tokens
    if specials is None:
        specials = []
    if special_first:
        sorted_tokens = specials + [token for token, freq in counter.items() if freq >= min_freq]
    else:
        sorted_tokens = [token for token, freq in counter.items() if freq >= min_freq] + specials

    # Create vocabulary
    return Vocab(OrderedDict((tok, 1) for tok in sorted_tokens))


