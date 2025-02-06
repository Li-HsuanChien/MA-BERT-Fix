from maa_datasets.utils import Data, _truncate_and_pad, build_vocab
from cfgs.constants import DATASET_MAP, DATASET_PATH_MAP
from torch.utils.data import DataLoader
import torch
import os

import re

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
import math

inf = math.inf
nan = math.nan
string_classes = (str, bytes)
int_classes = int
import collections.abc

container_abcs = collections.abc
FileNotFoundError = FileNotFoundError

def ensureDirs(*dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# def load_bert_sentences(config):
#     processor = DATASET_MAP[config.dataset]()
#     config.num_labels = processor.NUM_CLASSES

#     train_examples, dev_examples, test_examples = processor.get_sentences()

#     train_texts, train_labels, train_users, train_products, train_category, train_keywords = [], [], [], [], [], []
#     dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords =  [], [], [], [], [], []
#     test_texts, test_labels, test_users, test_products, test_category, test_keywords =  [], [], [], [], [], []
    
#     for example in train_examples:
#         train_texts.append(example.text)
#         train_labels.append(example.label)
#         train_users.append(example.user)
#         train_products.append(example.product)
#         train_category.append(example.category)
#         train_keywords.append(example.keywordlist)
#     for example in dev_examples:
#         dev_texts.append(example.text)
#         dev_labels.append(example.label)
#         dev_users.append(example.user)
#         dev_products.append(example.product)
#         dev_category.append(example.category)
#         dev_keywords.append(example.keywordlist)
#     for example in test_examples:
#         test_texts.append(example.text)
#         test_labels.append(example.label)
#         test_users.append(example.user)
#         test_products.append(example.product)
#         test_category.append(example.category)
#         test_keywords.append(example.keywordlist)
        
#     train_dataset = Data(train_texts, train_labels, train_users, train_products, train_category, train_keywords)
#     dev_dataset = Data(dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords)
#     test_dataset = Data(test_texts, test_labels, test_users, test_products, test_category, test_keywords)

#     train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
#     dev_dataloader = DataLoader(dev_dataset, batch_size=config.TEST.batch_size)
#     test_dataloader = DataLoader(test_dataset, batch_size=config.TEST.batch_size)

#     config.num_labels = processor.NUM_CLASSES

#     users, products, category = processor.get_attributes()
#     usr_stoi, prd_stoi, ctgy_stoi = load_attr_vocab(config.dataset, users, products, category)
#     config.num_usrs, config.num_prds, config.num_ctgy = len(usr_stoi), len(prd_stoi), len(ctgy_stoi)
    

    
#     keyword, keyword_counter = processor.get_keywords_and_counter()
#     keywordList, keyword_stoi = load_keywords(config.dataset, keyword, keyword_counter)
#     config.num_kws = len(keyword_stoi)

#     config.TRAIN.num_train_optimization_steps = int(
#         len(
#             train_examples) / config.TRAIN.batch_size / config.TRAIN.gradient_accumulation_steps) * config.TRAIN.max_epoch

#     return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi, ctgy_stoi, keywordList, keyword_stoi


# def load_bert_documents(config):
#     print("=== loading maa_datasets...")
#     processor = DATASET_MAP[config.dataset]()
#     config.num_labels = processor.NUM_CLASSES

#     train_examples, dev_examples, test_examples = processor.get_documents()

#     train_texts, train_labels, train_users, train_products, train_category, train_keywords = [], [], [], [], [], []
#     dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords =  [], [], [], [], [], []
#     test_texts, test_labels, test_users, test_products, test_category, test_keywords =  [], [], [], [], [], []

#     for example in train_examples:
#         train_texts.append(example.text)
#         train_labels.append(example.label)
#         train_users.append(example.user)
#         train_products.append(example.product)
#         train_category.append(example.category)
#         train_keywords.append(example.keywordlist)
#     for example in dev_examples:
#         dev_texts.append(example.text)
#         dev_labels.append(example.label)
#         dev_users.append(example.user)
#         dev_products.append(example.product)
#         dev_category.append(example.category)
#         dev_keywords.append(example.keywordlist)
#     for example in test_examples:
#         test_texts.append(example.text)
#         test_labels.append(example.label)
#         test_users.append(example.user)
#         test_products.append(example.product)
#         test_category.append(example.category)
#         test_keywords.append(example.keywordlist)

#     # train_texts, train_labels = [example.text for example in train_examples], [example.label for example in train_examples]
#     # dev_texts, dev_labels = [example.text for example in dev_examples], [example.label for example in dev_examples]
#     # test_texts, test_labels = [example.text for example in test_examples], [example.label for example in test_examples]

#     train_dataset = Data(train_texts, train_labels, train_users, train_products, train_category, train_keywords)
#     dev_dataset = Data(dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords)
#     test_dataset = Data(test_texts, test_labels, test_users, test_products, test_category, test_keywords)

#     train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
#     dev_dataloader = DataLoader(dev_dataset, batch_size=config.TEST.batch_size)
#     test_dataloader = DataLoader(test_dataset, batch_size=config.TEST.batch_size)

#     users, products, category = processor.get_attributes()
#     usr_stoi, prd_stoi, ctgy_stoi = load_attr_vocab(config.dataset, users, products, category)
#     config.num_usrs, config.num_prds, config.num_ctgy = len(usr_stoi), len(prd_stoi), len(ctgy_stoi)
#     config.num_labels = processor.NUM_CLASSES


#     keyword, keyword_counter = processor.get_keywords_and_counter()
#     keywordList, keyword_stoi = load_keywords(config.dataset, keyword, keyword_counter)
#     config.num_kws = len(keyword_stoi)
    
#     print("Done!")
#     config.TRAIN.num_train_optimization_steps = int(
#         len(
#             train_examples) / config.TRAIN.batch_size / config.TRAIN.gradient_accumulation_steps) * config.TRAIN.max_epoch
#     return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi, ctgy_stoi, keywordList, keyword_stoi


def multi_acc(y, preds):
    preds = torch.argmax(torch.softmax(preds, dim=-1), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def multi_mse(y, preds):
    mse_loss = torch.nn.MSELoss()
    preds = torch.argmax(torch.softmax(preds, dim=-1), dim=1)
    return mse_loss(y.float(), preds.float())


def generate_over_tokenizer(text, tokenizer, max_length=256):
    t = tokenizer.batch_encode_plus(text, padding='max_length',
                                    max_length=max_length,
                                    truncation=True,
                                    )
    input_ids = torch.tensor(t["input_ids"])
    attention_mask = torch.tensor(t["attention_mask"])
    return input_ids, attention_mask


def processor4baseline(text, tokenizer, config):
    input_ids = []
    # <PAD> id is 100
    for document in text:
        tokens = tokenizer.tokenize(document)
        new_tokens = _truncate_and_pad(tokens, config.BASE.max_length - 2, config.BASE.strategy)
        input_id = tokenizer.convert_tokens_to_ids(new_tokens)
        input_ids.append(input_id)
    return torch.tensor(input_ids, dtype=torch.long)


def processor4baseline_over_one_example(text, tokenizer, config):
    # <PAD> id is 100
    tokens = tokenizer.tokenize(text)
    new_tokens = _truncate_and_pad(tokens, config.BASE.max_length - 2, config.BASE.strategy)
    input_id = tokenizer.convert_tokens_to_ids(new_tokens)
    return torch.tensor(input_id, dtype=torch.long)

def fill_keywordlist(keywordlist):
    if len(keywordlist) < 5:
        keywordlist.extend(['<PAD>'] * (5 - len(keywordlist)))
    elif len(keywordlist) > 5:
        keywordlist[:] = keywordlist[0:5]  
    return keywordlist


def save_vectors(path, vocab, field='usr'):
    
    # itos, stoi, vectors, dim
    try:
      data = vocab.get_itos(), vocab.get_stoi()
      torch.save(data, os.path.join(path, '{}.pt'.format(field)))
    except Exception as e:
        print(f"Error in save_vector: {e}")

def save_keyword_list(path, keywordlist, field='keywordList'):
    
    # list
    try:
      torch.save(keywordlist, os.path.join(path, '{}.pt'.format(field)))
    except Exception as e:
        print(f"Error in save_keywordList: {e}")

def load_vocab(path, field='usr'):
    # itos, stoi, vectors, dim
    return torch.load(os.path.join(path, '{}.pt'.format(field)))

def load_keyword_list(path, field='keywordList'):
    # list
    return torch.load(os.path.join(path, '{}.pt'.format(field)))

def load_baselines_datasets(path, field='train', strategy='tail'):
    return torch.load(os.path.join(path, '{}_{}.pt'.format(field, strategy)))


def load_attr_vocab(dataset, users, products, category):
   
    try:
        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='prd')
        ctgy_itos, ctgy_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='ctgy')
    except Exception as e:
        print(f"Error in load_attr_vocab: {e}")
        print(f"Rebuilding attr vocab")
        usr_vocab = build_vocab(users)
        prd_vocab = build_vocab(products)
        ctgy_vocab = build_vocab(category)
        save_vectors(DATASET_PATH_MAP[dataset], usr_vocab, field='usr')
        save_vectors(DATASET_PATH_MAP[dataset], prd_vocab, field='prd')
        save_vectors(DATASET_PATH_MAP[dataset], ctgy_vocab, field='ctgy')
        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='prd')
        ctgy_itos, ctgy_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='ctgy')
    return usr_stoi, prd_stoi, ctgy_stoi

def load_keywords(dataset, keyword_counter, pos_keyword_counter, neg_keyword_counter):
   
    try:
        keyword_itos, keyword_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='keywordFull')
        pos_keyword_itos, pos_keyword_counter_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='keywordPos')
        pos_keyword_itos, neg_keyword_counter_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='keywordNeg')
        
    except Exception as e:
        print(f"Error in load_attr_keyword: {e}")
        keyword_vocab = build_vocab(keyword_counter)
        pos_keyword_vocab = build_vocab(pos_keyword_counter)
        neg_keyword_vocab = build_vocab(neg_keyword_counter)
        
        save_vectors(DATASET_PATH_MAP[dataset], keyword_vocab, field='keywordFull')
        save_vectors(DATASET_PATH_MAP[dataset], pos_keyword_vocab, field='keywordPos')
        save_vectors(DATASET_PATH_MAP[dataset], neg_keyword_vocab, field='keywordNeg')
        
        keyword_itos, keyword_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='keywordFull')
        pos_keyword_itos, pos_keyword_counter_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='keywordPos')
        pos_keyword_itos, neg_keyword_counter_stoi = load_vocab(DATASET_PATH_MAP[dataset], field='keywordNeg')
    return keyword_stoi, pos_keyword_counter_stoi, neg_keyword_counter_stoi


# def load_document4baseline(config, tokenizer):
#     processor = DATASET_MAP[config.dataset]()
#     config.num_labels = processor.NUM_CLASSES

#     train_examples, dev_examples, test_examples = processor.get_documents()

#     train_texts, train_labels, train_users, train_products, train_category, train_keywords = [], [], [], [], [], []
#     dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords =  [], [], [], [], [], []
#     test_texts, test_labels, test_users, test_products, test_category, test_keywords =  [], [], [], [], [], []

#     for example in train_examples:
#         train_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
#         train_labels.append(example.label)
#         train_users.append(example.user)
#         train_products.append(example.product)
#         train_category.append(example.category)
#         train_keywords.append(example.keywordlist)
#     for example in dev_examples:
#         dev_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
#         dev_labels.append(example.label)
#         dev_users.append(example.user)
#         dev_products.append(example.product)
#         dev_category.append(example.category)
#         dev_keywords.append(example.keywordlist)
        
#     for example in test_examples:
#         test_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
#         test_labels.append(example.label)
#         test_users.append(example.user)
#         test_products.append(example.product)
#         test_category.append(example.category)
#         test_keywords.append(example.keywordlist)
        

#     train_dataset = Data(train_texts, train_labels, train_users, train_products, train_category, train_keywords)
#     dev_dataset = Data(dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords)
#     test_dataset = Data(test_texts, test_labels, test_users, test_products, test_category, test_keywords)
    
#     train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
#     dev_dataloader = DataLoader(dev_dataset, batch_size=config.TEST.batch_size)
#     test_dataloader = DataLoader(test_dataset, batch_size=config.TEST.batch_size)

#     users, products, category = processor.get_attributes()
#     usr_stoi, prd_stoi, ctgy_stoi = load_attr_vocab(config.dataset, users, products, category)
#     config.num_labels = processor.NUM_CLASSES
#     config.num_usrs = len(usr_stoi)
#     config.num_prds = len(prd_stoi)
#     config.num_ctgy = len(ctgy_stoi)
    
#     keyword, keyword_counter = processor.get_keywords_and_counter()
   
#     keywordList, keyword_stoi = load_keywords(config.dataset, keyword, keyword_counter)
#     config.num_kws = len(keyword_stoi)

#     return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi, ctgy_stoi, keywordList, keyword_stoi


def load_document4baseline_from_local(config):
    try:
        train_input_ids, train_labels, train_users, train_products, train_category, train_keywords = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='train', strategy=config.BASE.strategy)
        dev_input_ids, dev_labels, dev_users, dev_products, dev_category, dev_keywords = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='dev', strategy=config.BASE.strategy)
        test_input_ids, test_labels, test_users, test_products, test_category, test_keywords = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='test', strategy=config.BASE.strategy)

        processor = DATASET_MAP[config.dataset]()
        config.num_labels = processor.NUM_CLASSES
        
        train_dataset = Data(train_input_ids, train_labels, train_users, train_products, train_category, train_keywords)
        dev_dataset = Data(dev_input_ids, dev_labels, dev_users, dev_products, dev_category, dev_keywords)
        test_dataset = Data(test_input_ids, test_labels, test_users, test_products, test_category, test_keywords)
        
        train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.TEST.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=config.TEST.batch_size)
        

        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
        ctgy_itos, ctgy_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='ctgy')
        config.num_usrs = len(usr_stoi)
        config.num_prds = len(prd_stoi)
        config.num_ctgy = len(ctgy_stoi)
        
        keyword_itos, keyword_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='keywordFull')
        pos_keyword_itos, pos_keyword_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='keywordPos')
        neg_keyword_itos, neg_keyword_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='keywordNeg')
        
        config.num_kws = len(keyword_stoi)
        config.num_negkws = len(neg_keyword_stoi)
        config.num_poskws = len(pos_keyword_stoi)
        
        
        config.TRAIN.num_train_optimization_steps = int(
            len(
                train_dataset) / config.TRAIN.batch_size / config.TRAIN.gradient_accumulation_steps) * config.TRAIN.max_epoch
        print("===loading {} document from local...".format(config.BASE.strategy))
        print("Done!")
        return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi, ctgy_stoi, keyword_itos, pos_keyword_itos, neg_keyword_itos
    except Exception as e:
        print(f"Error in load_document4baseline_from_local: {e}")
        
        save_datasets(config)
        
        train_input_ids, train_labels, train_users, train_products, train_category, train_keywords = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='train', strategy=config.BASE.strategy)
        dev_input_ids, dev_labels, dev_users, dev_products, dev_category, dev_keywords = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='dev', strategy=config.BASE.strategy)
        test_input_ids, test_labels, test_users, test_products, test_category, test_keywords = load_baselines_datasets(
            DATASET_PATH_MAP[config.dataset], field='test', strategy=config.BASE.strategy)

        processor = DATASET_MAP[config.dataset]()
        config.num_labels = processor.NUM_CLASSES

        train_dataset = Data(train_input_ids, train_labels, train_users, train_products, train_category, train_keywords)
        dev_dataset = Data(dev_input_ids, dev_labels, dev_users, dev_products, dev_category, dev_keywords)
        test_dataset = Data(test_input_ids, test_labels, test_users, test_products, test_category, test_keywords)
        
        train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.TEST.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=config.TEST.batch_size)
        
        usr_itos, usr_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='usr')
        prd_itos, prd_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='prd')
        ctgy_itos, ctgy_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='ctgy')
        config.num_usrs = len(usr_stoi)
        config.num_prds = len(prd_stoi)
        config.num_ctgy = len(ctgy_stoi)
        
        keyword_itos, keyword_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='keywordFull')
        pos_keyword_itos, pos_keyword_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='keywordPos')
        neg_keyword_itos, neg_keyword_stoi = load_vocab(DATASET_PATH_MAP[config.dataset], field='keywordNeg')
        config.num_kws = len(keyword_stoi)
        config.num_negkws = len(neg_keyword_stoi)
        config.num_poskws = len(pos_keyword_stoi)
        
        
        config.TRAIN.num_train_optimization_steps = int(
            len(
                train_dataset) / config.TRAIN.batch_size / config.TRAIN.gradient_accumulation_steps) * config.TRAIN.max_epoch
        print("===loading {} document from local...".format(config.BASE.strategy))
        print("Done!")
        return train_dataloader, dev_dataloader, test_dataloader, usr_stoi, prd_stoi, ctgy_stoi, keyword_itos, pos_keyword_itos, neg_keyword_itos


def save_datasets(config):
    
    try:
        from transformers import BertTokenizer
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        processor = DATASET_MAP[config.dataset]()
        train_examples, dev_examples, test_examples = processor.get_documents()
        
        train_texts, train_labels, train_users, train_products, train_category, train_keywords = [], [], [], [], [], []
        dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords = [], [], [], [], [], []
        test_texts, test_labels, test_users, test_products, test_category, test_keywords = [], [], [], [], [], []

        users, products, category = processor.get_attributes()
        usr_stoi, prd_stoi, ctgy_stoi = load_attr_vocab(config.dataset, users, products, category)
        config.num_labels = processor.NUM_CLASSES
        config.num_usrs = len(usr_stoi)
        config.num_prds = len(prd_stoi)
        config.num_ctgy = len(ctgy_stoi)

        keyword_counter = processor.get_keywords()
        pos_keyword_counter, neg_keyword_counter = processor.get_polarzied_keywords()
        keyword_stoi, pos_keyword_stoi, neg_keyword_stoi = load_keywords(config.dataset, keyword_counter, pos_keyword_counter, neg_keyword_counter)
        config.num_kws = len(keyword_stoi)

        
        print("==loading train maa_datasets")
        for step, example in enumerate(train_examples):
            train_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
            train_labels.append(example.label)
            train_users.append(example.user)
            train_products.append(example.product)
            train_category.append(example.category)
            train_keywords.append(fill_keywordlist(example.keywordlist))
            print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(train_examples),
                                                                step / len(train_examples) * 100),
                    end="")
        print("\rDone!".ljust(60))
        print("==loading dev maa_datasets")
        for step, example in enumerate(dev_examples):
            dev_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
            dev_labels.append(example.label)
            dev_users.append(example.user)
            dev_products.append(example.product)
            dev_category.append(example.category)
            dev_keywords.append(fill_keywordlist(example.keywordlist))
            
            print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dev_examples),
                                                                step / len(dev_examples) * 100),
                    end="")
        print("\rDone!".ljust(60))
        print("==loading test maa_datasets")
        for step, example in enumerate(test_examples):
            test_texts.append(processor4baseline_over_one_example(example.text, tokenizer, config))
            test_labels.append(example.label)
            test_users.append(example.user)
            test_products.append(example.product)
            test_category.append(example.category)
            test_keywords.append(fill_keywordlist(example.keywordlist))
            
            print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(test_examples),
                                                                step / len(test_examples) * 100),
                    end="")
        print("\rDone!".ljust(60))

        
        train_data = train_texts, train_labels, train_users, train_products, train_category, train_keywords
        dev_data = dev_texts, dev_labels, dev_users, dev_products, dev_category, dev_keywords
        test_data = test_texts, test_labels, test_users, test_products, test_category, test_keywords
        torch.save(train_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'train_{}.pt'.format(config.BASE.strategy)))
        torch.save(dev_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'dev_{}.pt'.format(config.BASE.strategy)))
        torch.save(test_data, os.path.join(DATASET_PATH_MAP[config.dataset], 'test_{}.pt'.format(config.BASE.strategy)))

        
    except Exception as e:
        print(f"Error in save_datasets: {e}")
        print("Full traceback:")
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        raise

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        max_length = 0
        for b in batch:
            if max_length < len(b): max_length = len(b)
        new_batch = []
        for b in batch:
            new_batch.append(torch.cat([b, torch.zeros(max_length - len(b), 768)]))
        out = None
        return torch.stack(new_batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
