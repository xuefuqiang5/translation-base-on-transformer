import argparse
import time
import torch 
import torch.nn.functional as F
from Process import build_dataset, read_data

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_largers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    opt = parser.parse_args()
    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
    if opt.device == 'cuda': 
        assert torch.cuda.is_available()
    print(opt) 
    read_data(opt) 
    train_iter, src_vocab, trg_vocab = build_dataset(opt)
    for batch_idx, batch in enumerate(train_iter):
        print(f"批次 {batch_idx}:")
        print(f"批次大小: {len(batch)}") 

if __name__ == "__main__":
    main()