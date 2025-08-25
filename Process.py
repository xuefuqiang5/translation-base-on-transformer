import pandas as pd
import torch
import numpy as np
import dill as pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from Tokenize import tokenize


def read_data(opt): 
    
    if opt.src_data is not None: 
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except: 
            print("error: '" + opt.src_data + "' file not fount") 
            quit()

    if opt.trg_data is not None: 
        try: 
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "'file not fount") 
            quit()

def build_vocab(opt): 

    def yield_tokens(data, is_src): 
        for s in data: 
            yield tokenize(opt.src_lang if is_src == True else opt.trg_lang).tokenizer(s)

    src_vocab = build_vocab_from_iterator(yield_tokens(opt.src_data, True), specials=[
        "<unk>", "<pad>", "<sos>", "<eos>"
    ])
    trg_vocab = build_vocab_from_iterator(yield_tokens(opt.trg_data, False), specials=[
        "<unk>", "<pad>", "<sos>", "<eos>"
    ])

    src_vocab.set_default_index(src_vocab["<unk>"])
    trg_vocab.set_default_index(trg_vocab["<unk>"])
    
    return src_vocab, trg_vocab

class TranslationDataset(Dataset): 

    def __init__(self, opt): 
        self.src_tokenizer = tokenize(opt.src_lang).tokenizer 
        self.trg_tokenizer = tokenize(opt.trg_lang).tokenizer
        self.src_text = opt.src_data
        self.trg_text = opt.trg_data
    
    def __len__(self): 
        
        return len(self.src_text)
    
    def __getitem__(self, idx): 

        raw_src_sentence, raw_trg_sentence = self.src_text[idx], self.trg_text[idx]
        return self.src_tokenizer(raw_src_sentence), self.trg_tokenizer(raw_trg_sentence)

def collate_fn(b, src_vocab, trg_vocab, opt): 
    src_batch, trg_batch = [], []
    for src, trg in b: 

        src_tensor = torch.tensor([src_vocab[token] for token in src], dtype=torch.long)
        trg_tensor = torch.tensor(
            [trg_vocab["<sos>"]] + [trg_vocab[token] for token in trg] + [trg_vocab["<eos>"]], dtype=torch.long
        )
        src_batch.append(src_tensor)
        trg_batch.append(trg_tensor)

    src_batch = pad_sequence(src_batch, padding_value=src_vocab["<pad>"], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=trg_vocab["<pad>"], batch_first=True)

    

    return src_batch.to(opt.device), trg_batch.to(opt.device)

def build_dataset(opt): 

    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs: 
        print('invalid src language: ' + opt.src_lang + 'supported language : ' + str(spacy_langs))
        quit()

    if opt.trg_lang not in spacy_langs:
        print('invalid src language: ' + opt.trg_lang + 'supported language : ' + str(spacy_langs))
        quit()

    print("loading spacy tokenizers...")

    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    raw_data = {'src' : [line for line in opt.src_data], 'trg' : [line for line in opt.trg_data]}

    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen) 

    df_flitered = df[mask]
    opt.src_data, opt.trg_data = df_flitered['src'].tolist(), df_flitered['trg'].tolist()

    dataset = TranslationDataset(opt)
    
    if opt.load_weights is None:
        src_vocab, trg_vocab = build_vocab(opt)

    train_iter = DataLoader(
        dataset, 
        batch_size=opt.batchsize,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab, trg_vocab, opt)
    )

    return train_iter, src_vocab, trg_vocab

def get_vocab(opt): 

    if opt.load_weights is not None:
        try: 
            print("loading presaved fields...")
            src_vocab = pickle.load(open(f'{opt.laod_weights}/SRC.pkl', 'rb'))
            trg_vocab = pickle.load(open(f'{opt.laod_weights}/TRG.pkl', 'rb'))
        except: 
            print("error openning SRC.pkl and txt.pkl files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return src_vocab, trg_vocab
