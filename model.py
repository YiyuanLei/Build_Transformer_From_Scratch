import torch
import torch.nn as nn
import math



class InputEmbeddings(nn.Module):
    # mapping between a number to a vector of d_model dimensions
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # max length of the sentences
        self.drop_out = nn.Dropout(drop_out)

        # create a matrix of dimension (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a position vector of dimension (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin and cos for position terms
        pe[:, 0::2] = torch.sin(position * div_term) # even has sin, ::2, start 0, up to N, go forward by 2, e.g. 0, 2, 4
        pe[:, 1::2] = torch.sin(position * div_term)  # odd has sin, 1::2, start 1, up to N, skip by 2, e.g. 1, 3, 5

        # add a dimension for batch sizes, dimension (1, seq_len, d_model), 1 is batch size
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += (self.pe[:, :x.shape[1], :]).requires_grad_(False) # makes pe not learned to save computation costs, x.shape[1] is the sequential length
        # slicing operation[dim1, dim2, dim3], [:,:x,:] means take all elements in dim1, take x length in dime2, and all in deim3
        return self.drop_out(x)
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps # numerical stability and avoid divided by zero standard deviation
        self.alpha = nn.Parameter(torch.ones(1)) # learnable multiplicative parameters
        self.bias = nn.Parameter(torch.zeros(1)) # learnable additive parameters
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim =  True) # keep dimensions easier for broadcasting
        std = x.std(dim = -1, keepdim = True) # dim = -1 , is x.ndim - 1 (last dimension)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model is not divisble by h"

        self.d_k = d_model //h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model) # output matrix
        # d_v and d_k at practically level is the same
        # we call d_v because of the softmax function transformation

    @staticmethod #static methods means you can call this class without a instance, can directly use the methods
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # use static method so that we can use this function without the instance of the class
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) --> # (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # @: matrix multiplication
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e-9)
            # when mask value is 0, the value will be replaced by a very small value, -1e-9
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # target value, and attention_scores used for visualizations
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d) -> (batch, seq_len, d)
        # multiply w_q with q
        key = self.w_v(k)
        value = self.w_v(v)

        # (batch, seq_len, d) -> (batch, seq_len, h, d_k) -> Through Transpose 1,2 -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # no need a self, because it's static method
        x, self.attention_scores  = MultiHeadAttentionBlock.attention(query, key, value, self.dropout)

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        # contiguous: pytorch to transform the shape and concat inplace
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # d_k*h is d_model,

        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        # multiply w_o with x
        return self.w_o(x)

class  ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__() # Always call parent constructor
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        # sublayer is the output of the next layer
        # the paper did sublayer first then apply normalization
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Moduel):
    def __init__(self, dropout: float, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock) -> None:
        super().__init__() # Always call parent constructor
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # why src_mask: mask applied to input of encoder so that embedding words won't interact with other words
        # ##multi-head attention part?
        # the first part is the multi_head attention (sublayer) within add_ norm (residual connections)
        x = self.residual_connections[0](x,  lambda x: self.self_attention_block(x, x, x, src_mask))
        # call the multi_head attention query, key, value, scr_mask
        # second part is the feedforward part
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, dropout: float, self_attention_block: MultiHeadAttentionBlock,cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # source_mask: encoder mask
        # tgt_mask: decoder mask

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # call the multi_head attention query, key, value, scr_mask
        # second part is the feedforward part
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    # Project embedding into the position of the vocabulary
    def __init__(self, d_model: int, vocab_size: int ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1) # (dim = - 1, to normalize the rows probability, ensures row sums to 1).

class Transformer(nn.Module):
    def __init(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,projection_layer: ProjectionLayer) -> None:
        # src_embed: input language
        # target_embed: target output languages
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgg_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    def project(self, x):
        return self.projection_layer(x)
class build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,  d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # create embedding layers
    src_embd = InputEmbeddings(d_model, src_vocab_size)
    tgt_embd = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encodings layers
    src_pos = PostionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_black = feed_forward_block(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_black, dropout)
        encoder_blocks.append(encoder_block)
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_dim, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_dim,  h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(docoder_self_attention_block, decoder_cross_attention, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection Lyaer

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #Create the transformer

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return transformer









