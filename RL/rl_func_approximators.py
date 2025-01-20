import torch

class qvalue_fn(torch.nn.Module):

    def __init__(self, qf_net) -> None:
        super(qvalue_fn,self).__init__()
        qf_net.eval()
        self.qf_net = qf_net

    def forward(self,x):
        output1 = self.qf_net(x)
        return output1
     
class actor_fn(torch.nn.Module):

    def __init__(self, latent_pi, mu) -> None:
        super(actor_fn,self).__init__()
        self.latent_pi = latent_pi
        self.mu = mu
        self.act_fn = torch.nn.Tanh()
        
    def forward(self,x):
        x = self.latent_pi(x)
        x = self.act_fn(self.mu(x))
        return x