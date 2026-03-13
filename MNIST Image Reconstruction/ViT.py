import torch
import torch.nn as nn
patch_size=16
colour_channels=3


class PatchEmbedding(nn.Module):
  def __init__(self, embed_dim, patch_size): #input dim: [b_s, no. of channels, 224, 224]
    super().__init__()
    self.conv_layer = nn.Conv2d(
        in_channels=colour_channels,
        out_channels = embed_dim,
        kernel_size = patch_size,
        stride = patch_size,
        padding = 0
    )
    self.flatten = nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x):
    #check that the input dimensions are compatible
    image_dim = x.shape[-1]
    assert image_dim % patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_dim}, patch size: {patch_size}"
    x = self.conv_layer(x)
    x = self.flatten(x)
    x = x.permute(0, 2, 1)
    return x
  
#coding a Multi-Head Attention Block

class MultiHeadAttn(nn.Module):
  def __init__(self, num_heads, embed_dim, num_patches, attn_dropout=0., proj_dropout=0.): #p argument for dropout must be a float
    super().__init__()
    self.num_heads = num_heads
    self.num_patches = num_patches
    self.embed_dim = embed_dim
    self.head_dim = embed_dim // num_heads
    self.qkv_proj = nn.Linear(embed_dim, embed_dim*3)
    self.norm_layer = nn.LayerNorm(embed_dim) #don't understand this
    self.scale = self.head_dim ** 0.5
    self.attn_dropout = nn.Dropout(attn_dropout)
    self.proj = nn.Linear(num_heads*self.head_dim, embed_dim) #ok but in this case uhh we purposely chose the head_dim to make sure the concatenated dimension is the same as the embedding dim so yes
    self.proj_drop = nn.Dropout(proj_dropout)


  def forward(self,x): #x.shape = [bs, 197, 768] -> [B, N, D]
    x_qkv = self.qkv_proj(x) #[bs, 197, 2304]
    x_qkv_reshaped = torch.reshape(x_qkv, (x.shape[0], self.num_patches+1, 3, self.num_heads, self.head_dim)).permute(2, 0, 3, 1, 4) #[3, 32, 12, 197, 64] ###
    q, k, v = x_qkv_reshaped[0], x_qkv_reshaped[1], x_qkv_reshaped[2] #each is now [32, 12, 197, 64] -> Q, K and V matrices for all 12 attention heads

    attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale #[32, 12, 197, 197]
    attn_scores = attn_scores.softmax(dim=-1)
    attn_scores = self.attn_dropout(attn_scores) #this attention dropout is applied to the attention weights
    attn_output = torch.matmul(attn_scores, v) #[32, 12, 197, 64] -> each row is an output context vector

    attn_output_concat = attn_output.permute(0, 2, 1, 3).flatten(start_dim=-2, end_dim=-1) #[32, 197, 768]

    z = self.proj(attn_output_concat) #[32, 197, 768]
    z = self.proj_drop(z)

    return z #[32, 197, 768]

#coding the MLP block
class MLP(nn.Module):
  def __init__(self, embed_dim, mlp_dim, act_fn=nn.GELU, dropout=0.):
    super().__init__()
    self.act_fn = act_fn() # note: if you do not include the brackets here, the GELU is not actly applied to x in line *
    self.linear1 = nn.Linear(embed_dim, mlp_dim)
    self.linear2 = nn.Linear(mlp_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x): #x.shape = [8, 197, 768]
    x = self.linear1(x)
    x = self.act_fn(x) # *
    #print(type(x)) -> returns nn.Modules activation function sth sth instead of torch.tensor as expected
    x = self.dropout(x)
    x = self.linear2(x)
    x = self.dropout(x)

    return x

#put together one Transformer encoder block
class TransformerEncoder(nn.Module):
  def __init__(self, num_heads, embed_dim, num_patches, mlp_dim, attn_dropout=0., proj_dropout=0., mlp_dropout=0., norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1 = norm_layer(embed_dim)
    self.norm2 = norm_layer(embed_dim)
    self.MHSA = MultiHeadAttn(num_heads, embed_dim, num_patches, attn_dropout, proj_dropout)
    self.MLP = MLP(embed_dim, mlp_dim, act_fn=nn.GELU, dropout=mlp_dropout)

  def forward(self, x):
    x = self.MHSA(self.norm1(x)) + x
    x = self.MLP(self.norm2(x)) + x

    return x
  
#Put together the Vision Transformer (modified for regression)
import torch.nn as nn

class ViT(nn.Module):
  def __init__(self, num_heads, num_patches, mlp_dim, patch_size=16,num_channels=3, num_encoders = 12, attn_dropout=0., proj_dropout=0., mlp_dropout=0., norm_layer=nn.LayerNorm):
    super().__init__()
    self.embed_dim = patch_size**2 * num_channels
    self.num_encoders = num_encoders
    self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim), requires_grad=True)
    self.pos_token = nn.Parameter(torch.randn(1, num_patches+1, self.embed_dim), requires_grad=True)
    self.patch_embedding = PatchEmbedding(self.embed_dim, patch_size)
    self.transformer_encoder = nn.Sequential(*[TransformerEncoder(num_heads, self.embed_dim, num_patches, mlp_dim, attn_dropout, proj_dropout, mlp_dropout, norm_layer) for _ in range(num_encoders)])
    self.regression_head = nn.Linear(self.embed_dim, 1)

  def forward(self, x):
    #create patch embeddings
    p_e = self.patch_embedding(x)
    cls_token=self.cls_token.expand(x.shape[0], -1, -1)
    pos_token=self.pos_token.expand(x.shape[0], -1, -1)
    patches_w_cls = torch.cat((cls_token, p_e), dim=1)
    patches_w_cls_n_pos = patches_w_cls + pos_token

    #send through the transformer encoder?
    x = self.transformer_encoder(patches_w_cls_n_pos) #output is [bs, 197, 768]

    #send through the MLP head
    cls_token_output = x[:,0,:] #[bs, 1, 768]
    pred=self.regression_head(cls_token_output)

    return pred
