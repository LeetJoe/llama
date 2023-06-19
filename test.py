



prompts = ["I believe the meaning of life is", "Simply put, the theory of relativity states that ", "Building a website can be done in 10 simple steps:\n"]
max_gen_len = 512
temperature = 0.8
top_p = 0.95
bsz = len(prompts)
from llama import Tokenizer
tokenizer = Tokenizer(model_path='/data1/neosong/models/llama/origin/tokenizer.model')


# 当 bos = True 时,输出的实向量开头会多一个 '1', 当 eos = True 时,输出的实向量末尾会多一个 '2'.
prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

# tokenizer.decode([1, 29871, 236, 169, 174, 236, 193, 155, 236, 166, 160, 236, 180, 190, 234, 132, 168, 236, 189, 167, 236, 160, 147, 236, 166, 144, 233, 159, 167, 233, 186, 191, 236, 169, 174, 235, 176, 185])
# '馫龘飝鱻灥麤靐飍朤淼馫譶'

# prompt_tokens:
'''
[
    [1, 306, 4658, 278, 6593, 310, 2834, 338], 
    [1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871], 
    [1, 17166, 263, 4700, 508, 367, 2309, 297, 29871, 29896, 29900, 2560, 6576, 29901, 13]
]
'''

min_prompt_size = min([len(t) for t in prompt_tokens])   # 8
max_prompt_size = max([len(t) for t in prompt_tokens])   # 15
total_len = min(512, max_gen_len + max_prompt_size)      # 512

import torch
# tokens 是用来存放最终生成结果的. 初始全为-1
tokens = torch.full((bsz, total_len), tokenizer.pad_id).cuda().long()   # tokens.shape: torch.Size([3, 512])


# put prompt_tokens into tokens
for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t).long()

# mask of tokens, input_text_mask[x, y] = False if tokens[x, y] = -1.
input_text_mask = tokens != tokenizer.pad_id   # tokenizer.pad_id = -1



# 下面是为cuda分配任务以及初始化, 如果不执行下面这段,在实例化model = Transformer(model_args)的时候会报 "model parallel group is not initialized"
import os
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8976"
local_rank = 0
world_size = 1
torch.distributed.init_process_group("nccl")

initialize_model_parallel(world_size) # todo:

torch.cuda.set_device(local_rank)
torch.manual_seed(1)


# 这里是实例化model,此处实例化 7B.
import json
from pathlib import Path

# list of PosixPath, have only one path.
checkpoints = sorted(Path('/data1/neosong/models/llama/origin/7B').glob("*.pth"))
ckpt_path = checkpoints[0]

# 这里会把模型加载到内存  type of checkpoint is dict.
checkpoint = torch.load(ckpt_path, map_location="cpu")

'''
checkpoint.keys:
[
    'tok_embeddings.weight', 'norm.weight', 'output.weight', 
    
    'layers.0.attention.wq.weight', 'layers.0.attention.wk.weight', 'layers.0.attention.wv.weight', 'layers.0.attention.wo.weight', 
    'layers.0.feed_forward.w1.weight', 'layers.0.feed_forward.w2.weight', 'layers.0.feed_forward.w3.weight', 
    'layers.0.attention_norm.weight', 'layers.0.ffn_norm.weight', 
    ...
    'layers.31.attention.wq.weight', 'layers.31.attention.wk.weight', 'layers.31.attention.wv.weight', 'layers.31.attention.wo.weight', 
    'layers.31.feed_forward.w1.weight', 'layers.31.feed_forward.w2.weight', 'layers.31.feed_forward.w3.weight', 
    'layers.31.attention_norm.weight', 'layers.31.ffn_norm.weight', 
    
    # 下面这32个inner_attention在后面的 model = Transformer(model_args) 里面没有对应,应该会被 model 舍弃.
    'layers.0.attention.inner_attention.rope.freqs',
    ...
    'layers.31.attention.inner_attention.rope.freqs'
]
'''


with open(Path('/data1/neosong/models/llama/origin/7B') / "params.json", "r") as f:
    params = json.loads(f.read())
# print(params): {'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': -1}


from llama import ModelArgs
from llama import Transformer
model_args: ModelArgs = ModelArgs(max_seq_len=512, max_batch_size=32, **params)
# print(model_args): ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=-1, multiple_of=256, norm_eps=1e-06, max_batch_size=32, max_seq_len=512)


model_args.vocab_size = tokenizer.n_words   # 32000

torch.set_default_tensor_type(torch.cuda.HalfTensor)    # todo ?

# 此时的 model 是初始化的,里面都是空的.
model = Transformer(model_args)

'''
print(model): 与上面的checkpoint相对应.
Transformer(
  (tok_embeddings): ParallelEmbedding()
  (layers): ModuleList(
    (0-31): 32 x TransformerBlock(
      (attention): Attention(
        (wq): ColumnParallelLinear()
        (wk): ColumnParallelLinear()
        (wv): ColumnParallelLinear()
        (wo): RowParallelLinear()
      )
      (feed_forward): FeedForward(
        (w1): ColumnParallelLinear()
        (w2): RowParallelLinear()
        (w3): ColumnParallelLinear()
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (output): ColumnParallelLinear()
)
'''


torch.set_default_tensor_type(torch.FloatTensor)  # todo ?


# 此时 model 加载了 checkpoint, 实例化完成. 但是其中未包含 checkpoint 里的后 32 层 inner_attention.
model.load_state_dict(checkpoint, strict=False)


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token



start_pos = min_prompt_size  # 8
prev_pos = 0

cur_pos = start_pos


# 此处循环为了使用批量 generation, 要以最短的 input 为准; 这样对于较长的 input 来说, 前面一些 generation 是无用功, 虽然会 generate 但会直接舍弃, 直到此 input 的结尾处为止.
for cur_pos in range(start_pos, total_len):

    # tokens[:, prev_pos:cur_pos] 截取了 tokens 左侧都不为 -1 的最大矩阵块
    # 在 forward 内部, 使用 model.to_embeddings() 将 tokens 转化成实数矩阵.
    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)    # logits.shape: torch.Size([3, 32000])

    # 预测得到的下一个token;
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)    # 输出的probs是固定了,也即已经包含了所有可能的结果.
        # 取前5%的结果, 多次使用 sample_top_p 得到的 next_token 可能会有不同.
        next_token = sample_top_p(probs, top_p)
    else:
        # 取惟一结果, 是一个三维(对应三个输入)向量,每一个值表示对其中一个输入的 next word 的预测.
        next_token = torch.argmax(logits, dim=-1)    # print(next_token): tensor([  304, 14675,   263], device='cuda:0'), 'to', 'evolution', 'a'

    # reshape(-1) 表示把 next_token 打平,即如果 next_token 是一个 3 * 2 的矩阵, reshape(-1) 后会变成一个 1 * 6 的向量.
    next_token = next_token.reshape(-1)

    # only replace token if prompt has already been generated
    # 因为生成是所有输入同时进行的,所以要以最短的输入作为超始位置; 生成之后,如果下一个位置原prompt是-1,才把生成的内容填上去;
    # 若对应位置的下一个位置原 prompt 尚未结束,则舍弃对应位置的生成内容,使用原来 prompt 的内容;
    next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    )
    print(next_token)
    tokens[:, cur_pos] = next_token
    prev_pos = cur_pos



