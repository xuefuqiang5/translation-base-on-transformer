import pandas as pd
import torch
from torch.utils.data import Dataset, dataloader
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

def build_dataset