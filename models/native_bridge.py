"""Discrete Brownian DSB with alternating learned transition increments.

DSB's regression is the mean_match=False form of the author's 2D recipe:
MSE(net(x_new, T-t_next), transition_mean(x_old)-transition_mean(x_new)).
Each half-pass freezes the opposite direction. Endpoints are independently
fixed: standard Gaussian prior and supplied data. The initial reference has
zero drift (Brownian), as required by the paper's Brownian LID interface.
Trajectories are resampled rather than persisted in a cache, so checkpoint
resume depends only on the recorded RNG and step, not a hidden trajectory set.
"""
import copy
import torch

from models.native_tasks import NativeField, _condition


class NativeDSB(NativeField):
    def __init__(self, architecture, family, cfg):
        super().__init__(architecture, family, cfg)
        self.backward_core = copy.deepcopy(self.core)
        self.register_buffer('ipf_training_step', torch.zeros((),dtype=torch.long))

    def set_training_step(self, step):
        old_phase = self.phase()
        self.ipf_training_step.fill_(step)
        changed = self.phase() != old_phase
        if changed:
            core = self.core if self.phase()%2 == 0 else self.backward_core
            # The released 2D recipe uses use_prev_net=False. Reset only the
            # direction being fitted; its opposite remains the fixed teacher.
            for layer in core.modules():
                if hasattr(layer,'reset_parameters'):
                    layer.reset_parameters()
            torch.nn.init.zeros_(core.output.weight)
            torch.nn.init.zeros_(core.output.bias)
        return changed

    def phase(self):
        cfg = self.training_config
        per_pass = cfg.steps//(2*cfg.dsb_ipf_rounds)
        return min((max(1,int(self.ipf_training_step))-1)//per_pass,2*cfg.dsb_ipf_rounds-1)

    def increment(self, x, time, *, dataward):
        return self.raw(x,time,self.core if dataward else self.backward_core)

    def forward(self, x, time):
        """Forward drift toward the data; raw network output is dt * drift."""
        return self.increment(x,time,dataward=True)/self.training_config.dsb_step_size

    def canonical_posterior(self, x, scale):
        lam = _condition(scale,x)
        cfg = self.training_config
        tau = lam.square()/2
        t = cfg.dsb_num_steps*cfg.dsb_step_size-tau
        return x+tau[:,None]*self(x,t)

    def regression_batch(self, terminal_data, generator):
        cfg = self.training_config
        phase = self.phase(); dataward = phase%2 == 0
        dt = cfg.dsb_step_size; horizon = cfg.dsb_num_steps*dt
        with torch.no_grad():
            x = (terminal_data.detach().clone() if dataward else
                 torch.randn(terminal_data.shape,device=terminal_data.device,
                             dtype=terminal_data.dtype,generator=generator))
            index = torch.randint(cfg.dsb_num_steps,(len(x),),device=x.device,generator=generator)
            observed = torch.empty_like(x); target = torch.empty_like(x)
            condition = torch.empty(len(x),device=x.device,dtype=x.dtype)
            for k in range(cfg.dsb_num_steps):
                t = torch.full((len(x),),k*dt,device=x.device,dtype=x.dtype)
                # Only the first IPF projection samples the reference Brownian
                # process. Later passes use the learned opposite transition.
                old_mean = x if phase == 0 else x+self.increment(x,t,dataward=not dataward)
                z = torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
                new = old_mean+(2*dt)**.5*z
                new_mean = new if phase == 0 else new+self.increment(new,t,dataward=not dataward)
                choose = index == k
                observed[choose] = new[choose]
                target[choose] = (old_mean-new_mean)[choose]
                condition[choose] = horizon-(k+1)*dt
                x = new
        return observed,condition,target,dataward

    def ipf_loss(self, terminal_data, generator):
        x,t,target,dataward = self.regression_batch(terminal_data,generator)
        return (self.increment(x,t,dataward=dataward)-target).square().mean()
