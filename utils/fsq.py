import torch
import torch.nn as nn

def round_ste(z):
    zhat = z.round()
    return z + (zhat - z).detach()

class FSQ(nn.Module):
    def __init__(self, dim=96):
        super(FSQ, self).__init__()

        self.level_list = [8,8,8,6,5]

        levels = torch.tensor(self.level_list, dtype=torch.int32)
        self.register_buffer("levels", levels, persistent = False)
        basis = torch.cumprod(torch.tensor([1] + self.level_list[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("basis", basis, persistent = False)

        self.dim = dim
        self.codebook_dim = len(self.level_list)

        self.project_in = nn.Linear(self.dim, self.codebook_dim)
        self.project_out = nn.Linear(self.codebook_dim, self.dim)

    def bound(self, z, eps=1e-3):
        half_l = (self.levels - 1) * (1 - eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset
    
    def quantize(self, z):
        quantized = round_ste(self.bound(z))
        half_width = self.levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat):
        zhat = self._scale_and_shift(zhat)
        return (zhat * self.basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices):
        indices = torch.unsqueeze(indices, dim=-1)
        codes_non_centered = (indices // self.basis) % self.levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        codes = self.project_out(codes)

        return codes

    def forward(self, z):
        z = self.project_in(z)
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        out = self.project_out(codes)

        return out, indices
    
if __name__ == '__main__':
    quantizer = FSQ()

    x = torch.randn(size=(5, 8, 96))

    xhat, indices = quantizer(x)

    print(xhat.shape)
    print(torch.sum(xhat, dim=-1))
    print(indices.shape)
    print(indices)

    xhat_ = quantizer.indices_to_codes(indices)
    
    print(xhat_.shape)
    print(torch.sum(xhat_, dim=-1))



    
