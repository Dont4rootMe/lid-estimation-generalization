import copy
import torch
from models.noise_pairing import paired_corruption
from models.training import diffusion_ve_dsm_loss


class Field(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bias=torch.nn.Parameter(torch.tensor([.2,-.1],dtype=torch.float64))
    def forward(self,x,s):return torch.sin(x)+self.bias


def test_pair_marginals_and_untouched_validation():
    model=Field();model._lid_noise_pairing='antithetic_v1'
    x=torch.randn(8,2,dtype=torch.float64);s=torch.rand(8,dtype=torch.float64);z=torch.randn_like(x)
    a,b,c=paired_corruption(model,x,s,z)
    assert torch.equal(a[:4],x[:4]) and torch.equal(a[4:],x[:4])
    assert torch.equal(b[:4],s[:4]) and torch.equal(b[4:],s[:4])
    assert torch.equal(c[:4],z[:4]) and torch.equal(c[4:],-z[:4])
    model.eval();a,b,c=paired_corruption(model,x,s,z)
    assert a is x and b is s and c is z


def test_native_antithetic_loss_and_gradient_equals_two_signed_corruptions():
    x=torch.randn(8,2,dtype=torch.float64);model=Field();model._lid_noise_pairing='antithetic_v1'
    g=torch.Generator().manual_seed(77);reference_generator=torch.Generator().manual_seed(77)
    loss=diffusion_ve_dsm_loss(model,x,sigma_min=.01,sigma_max=64.,generator=g)
    u=torch.rand(8,dtype=x.dtype,generator=reference_generator)
    s=torch.exp(torch.log(torch.tensor(.01,dtype=x.dtype))+u*torch.log(torch.tensor(6400.,dtype=x.dtype)))
    z=torch.randn(x.shape,dtype=x.dtype,generator=reference_generator)
    terms=[((model(x[:4]+sign*s[:4,None]*z[:4],s[:4])-x[:4])/s[:4,None]).square().mean() for sign in [-1,1]]
    expected=sum(terms)/2
    torch.testing.assert_close(loss,expected,rtol=1e-12,atol=1e-12)
    actual_gradient=torch.autograd.grad(loss,model.bias,retain_graph=True)[0]
    expected_gradient=torch.autograd.grad(expected,model.bias)[0]
    torch.testing.assert_close(actual_gradient,expected_gradient,rtol=1e-12,atol=1e-12)
    assert torch.equal(g.get_state(),reference_generator.get_state())
