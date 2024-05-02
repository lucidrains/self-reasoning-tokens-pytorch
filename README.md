<img src="./reasoning-tokens.png" width="400px"></img>

## Self Reasoning Tokens - Pytorch (wip)

Exploration into the proposed <a href="https://reasoning-tokens.ghost.io/reasoning-tokens/">Self Reasoning Tokens</a> by Felipe Bonetto. The blog post seems a bit unfleshed out, but the idea of stop gradients from next token(s) is an interesting one.

My initial thought was to apply a stop gradient mask on the attention matrix, but then realized that the values of the "reasoning" tokens could not be stop gradiented correctly without memory issues.

While walking the dog and meditating on this, I came to the realization that one can create independent stop gradient masks for queries, keys, values in either flash attention or a custom attention backwards, and there may be a whole array of possibilities there. If any experiments come back positive from this exploration, will build out a concrete implementation of this.

## Citations

```bibtex
@misc{Bonetto2023,
    author  = {Felipe Bonetto},
    url     = {https://reasoning-tokens.ghost.io/reasoning-tokens/}
}
```
