import torch
import numpy as np
from torch.autograd import Variable

def nopeak_mask(size, opt): 
    
    np_mask = np.triu(np.ones(1, size, size), k=1).astype('uint8') 
    np_mask = Variable(torch.from_numpy(np_mask == False).to(opt.device))

    return np_mask

def create_masks(src, trg, opt): 
    src_mask = (src != opt.src_pad).unsqueeze(-2).to(opt.deivce)

    if trg is not None: 
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2).to(opt.device)
        size = trg.size(1)
        np_mask = nopeak_mask(size, opt)
        trg_mask = trg_mask & np_mask
    
    else: 
        trg_mask = None
    
    return src_mask, trg_mask 