import torch
import torch.nn as nn


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    @torch.no_grad()
    def encode(self, input_acts, indices=None):
        if indices is not None:
            pre_acts = input_acts @ self.W_enc[:,indices] + self.b_enc[indices]
            mask = (pre_acts > self.threshold[indices])
            acts = mask * torch.nn.functional.relu(pre_acts)
            return acts
        
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    @torch.no_grad()
    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    @torch.no_grad()
    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

import numpy as np

nickname_to_repo_id = {
    'res': "google/gemma-scope-2b-pt-res",
    'att': 'google/gemma-scope-2b-pt-att',
    'mlp': 'google/gemma-scope-2b-pt-mlp',
    'transcoders': 'google/gemma-scope-2b-pt-transcoders',
    'res-2b': "google/gemma-scope-2b-pt-res",
    'att-2b': 'google/gemma-scope-2b-pt-att',
    'mlp-2b': 'google/gemma-scope-2b-pt-mlp',
    'transcoders-2b': 'google/gemma-scope-2b-pt-transcoders',
    'res-9b': "google/gemma-scope-9b-pt-res",
    'att-9b': 'google/gemma-scope-9b-pt-att',
    'mlp-9b': 'google/gemma-scope-9b-pt-mlp',
    'res-27b': 'google/gemma-scope-27b-pt-res'
}

from huggingface_hub import HfFileSystem, hf_hub_download
import numpy as np
import torch
# from utils import JumpReLUSAE

def load_gemma_sae(type, layer, l0, width='65k'):
    global lm
    assert type in {'mlp', 'att', 'res', 'transcoders'}
    repo_id = f'google/gemma-scope-2b-pt-{type}'
    sae_loc = f'layer_{layer}/width_{width}'
    
    fs = HfFileSystem()
    sae_names = [f['name'].split(repo_id)[1][1:] for f in fs.ls(repo_id + '/' + sae_loc)]
    
    
    l0s = [int(name.split('_')[-1]) for name in sae_names]
    
    closest_l0_index = np.abs(np.log(np.array(l0s)) - np.log(l0)).argmin()
    
    print(f'Retrieved SAE: gemma-2-2b/{type}/{sae_names[closest_l0_index]}')
        
    sae_dir = sae_names[closest_l0_index]
    
    path_to_params = hf_hub_download(
        repo_id = repo_id, # e.g., "gpt2-small-res-jb". See other options in https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml
        filename = sae_dir + "/params.npz", # e.g., "blocks.8.hook_resid_pre". Won't always be a hook point
        force_download=False,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(lm.device) for k, v in params.items()}


    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)

    return sae

from noa_tools import register_hook, remove_hooks

class CustomBreakError(ValueError):
    pass

# def cache_hook(m, i, o):
#     assert isinstance(o, tuple)
#     m.cache['out'] = o[0]

def cache_inp_and_break_hook(m, i, o):
    if isinstance(i, tuple):
        i = i[0]
    m.cache['sae_inp'] = i
    raise CustomBreakError("Intentional break in forward pass")

def cache_out_and_break_hook(m, i, o):
    if isinstance(o, tuple):
        o = o[0]
    m.cache['sae_inp'] = o
    raise CustomBreakError("Intentional break in forward pass")
  

def register_sae(lm, type, layer, l0, width='65k'):
    '''
    - Removes existing language model hooks
    - Registers a hook on the module that the SAE is reading from. The hook caches sae inputs in module.cache['sae_inp']
    - Returns module, sae
    '''
    assert type in {'mlp', 'att', 'res'}
    
    remove_hooks(lm)
    
    sae = load_gemma_sae(type=type, layer=layer, width=width, l0=l0).to(dtype=lm.dtype, device=lm.device)
    block = lm.model.layers[layer]

    if type == 'mlp':
        module = block.mlp
        register_hook(module, cache_out_and_break_hook)
    elif type == 'att':
        module = block.self_attn.o_proj
        register_hook(module, cache_inp_and_break_hook)
    elif type == 'res':
        module = block
        register_hook(module, cache_out_and_break_hook)
    else:
        raise ValueError(f'Unexpected sae type found: {type}')

    return module, sae
        
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import CustomBreakError, register_sae
from noa_tools import clear_cache
from einops import rearrange

from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from nnsight import NNsight
from utils import load_gemma_sae




tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
lm = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")#.to(torch.bfloat16).cuda()

import numpy as np

# tok_strs = np.array([tokenizer.decode([tok_id]).replace(' ', '·').replace('\n', '⤶') for tok_id in range(256000)])
tok_strs = np.array([tokenizer.decode([tok_id]) for tok_id in range(256000)])

@torch.no_grad()
def get_acts(layer, l0, sae_type, width='16k', batch_size=200, indices=range(0,100), num_docs=10_000):
    global lm

    num_batches= num_docs // batch_size
    dataloader = DataLoader(ds['train'], batch_size=batch_size)

    clear_cache(lm)
    module, sae = register_sae(lm, layer=layer, l0=l0, type=sae_type, width=width)


    all_acts = []
    all_toks = []
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        out = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok_ids, attn_mask = out['input_ids'], out['attention_mask']
        tok_ids = tok_ids[attn_mask[:,0].bool()].to(lm.device)
        
        try:
            lm.forward(tok_ids)
        except CustomBreakError:
            pass
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        sae_inp = module.cache['sae_inp']
        
        acts = sae.encode(sae_inp, indices=indices)

        all_acts.append(acts)
        all_toks.append(tok_ids)

        if len(all_acts) > num_batches:
            break

    acts = torch.cat(all_acts, dim=0).cpu()
    toks = tok_strs[np.concatenate([toks.cpu().numpy() for toks in all_toks], axis=0)]

    
    acts = acts/(rearrange(acts, 'b s a -> ( b s ) a').max(dim=0).values[None,None]+1e-10)
    
    acts = rearrange(acts, 'b s a -> a b s')
    return acts, toks

import pysvelte

@torch.no_grad()
def see_acts(acts, toks, **kwargs):
    kwargs['start'] = kwargs.get('start', 0.8)
    kwargs['aggr'] = kwargs.get('aggr', 'signed_absmax')
    
    assert len(acts.shape) == 2
    assert len(toks.shape) == 2
    
    feat_mask = acts.max(dim=-1).values > 0
    docs = toks[feat_mask].tolist()
    acts = acts[feat_mask].cpu().tolist()
    pysvelte.WeightedDocs(docs=docs, acts=acts,**kwargs).show()

# def load_gemma_sae(repo_id="google/gemma-scope-2b-pt-res", filename="layer_10/width_16k/average_l0_77"):
#     from huggingface_hub import hf_hub_download
    
#     if repo_id in nickname_to_repo_id:
#         repo_id = nickname_to_repo_id[repo_id]
    

#     path_to_params = hf_hub_download(
#         repo_id = repo_id, # e.g., "gpt2-small-res-jb". See other options in https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml
#         filename = filename + "/params.npz", # e.g., "blocks.8.hook_resid_pre". Won't always be a hook point
#         force_download=False,
#     )

#     params = np.load(path_to_params)
#     pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}


#     sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
#     sae.load_state_dict(pt_params)

#     return sae