import torch
from torch.autograd.function import Function

from einops import einsum, rearrange

def exists(val):
    return val is not None

# custom function

class StopGraddableAttentionFunction(Function):

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        q,
        k,
        v,
        mask,
        attn_mask,
        causal: bool,
        q_stop_grad_mask,
        k_stop_grad_mask,
        v_stop_grad_mask,
    ):
        scale = q.shape[-1] ** -0.5

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * scale

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(col_mask, 'b j -> b 1 1 j')
            sim.masked_fill_(~mask, max_neg_value)

        if exists(attn_mask):
            sim.masked_fill_(~attn_mask, max_neg_value)

        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        ctx.args = (
            causal,
            scale,
            mask,
            q_stop_grad_mask,
            k_stop_grad_mask,
            v_stop_grad_mask
        )

        ctx.save_for_backward(
            q, k, v,
            attn,
            out
        )

        return out

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):

        (
            causal,
            scale,
            mask,
            q_stop_grad_mask,
            k_stop_grad_mask,
            v_stop_grad_mask
        ) = ctx.args

        q, k, v, p, o = ctx.saved_tensors

        # stop grad masks are either type bool, with True indicating stop grad, or can be type float, in which case it will scale the gradients

        if q_stop_grad_mask.dtype == torch.bool:
            q_stop_grad_mask = (~q_stop_grad_mask).float()

        if k_stop_grad_mask.dtype == torch.bool:
            k_stop_grad_mask = (~k_stop_grad_mask).float()

        print(v_stop_grad_mask.dtype)
        if v_stop_grad_mask.dtype == torch.bool:
            print('hmmm')
            v_stop_grad_mask = (~v_stop_grad_mask).float()

        # softmax D

        D = (do * o).sum(dim = -1, keepdims = True)        

        # stop grad for values

        p_v = p

        if exists(v_stop_grad_mask):
            p_v.mul_(v_stop_grad_mask)

        # dv

        dv = einsum(p_v, do, 'b h i j, b h i d -> b h j d')

        # prep for dq and dk

        dp = einsum(do, v, 'b h i d, b h j d -> b h i j')
        ds = p * scale * (dp - D)

        # handle stop grad masking for queries and keys

        ds_q = ds_k = ds

        if exists(q_stop_grad_mask):
            ds_q.mul_(q_stop_grad_mask)

        if exists(k_stop_grad_mask):            
            ds_k.mul_(k_stop_grad_mask)

        # dq and dk

        dq = einsum(ds_q, k, 'b h i j, b h j d -> b h i d')
        dk = einsum(ds_k, q, 'b h i j, b h i d -> b h j d')

        return dq, dk, dv, None, None, None, None, None, None

# convenience method with defaults

stop_graddable_attn_ = StopGraddableAttentionFunction.apply

def stop_graddable_attn(
    q, k, v,
    mask = None,
    attn_mask = None,
    causal = False,
    q_stop_grad_mask = None,
    k_stop_grad_mask = None,
    v_stop_grad_mask = None
):
    return stop_graddable_attn_(q, k, v, mask, attn_mask, causal, q_stop_grad_mask, k_stop_grad_mask, v_stop_grad_mask)
