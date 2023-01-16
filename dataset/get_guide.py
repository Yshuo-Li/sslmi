def get_guide(tensor):
    g = tensor.sum(dim=(-2, -1))
    g = g/g.sum(dim=-1, keepdim=True)*g.shape[-1]/2
    return g[..., 0, :]
