from dataclasses import replace
import copy
import torch
import numpy as np
from models.training import TrainingConfig,train_model
from experiments.lambda_repair_train import extend_constant_progress
from experiments.lambda_repair_train import prepare_terminal_progress
from experiments.lambda_repair_train import replan_constant_progress
from experiments.lambda_repair_train import prepare_empirical_target_progress


def test_constant_budget_extension_matches_uninterrupted_training(tmp_path,monkeypatch):
    from models import training
    original_save=training._atomic_torch_save
    def retain_prefix(path,payload):
        original_save(path,payload)
        if path.name=='prefix_progress.pt':torch.save(payload,tmp_path/'retained_prefix.pt')
    monkeypatch.setattr(training,'_atomic_torch_save',retain_prefix)
    rng=np.random.default_rng(19);x=rng.normal(size=(96,6)).astype(np.float32)
    cfg=TrainingConfig(field_hidden_sizes=(24,16,8),time_embedding_dim=128,device='cpu',
        training_mode='fixed_steps_v1',steps=4,warmup_steps=0,validation_interval_steps=2,
        sigma_min=.01,sigma_max=64.,batch_size=16,early_stopping_patience=None,
        field_preconditioning='covariance_span_v1',noise_pairing='antithetic_v1')
    train_model('diffusion',x[:80],x[80:],cfg,tmp_path/'prefix.pt',progress_checkpoint_path=tmp_path/'prefix_progress.pt')
    payload=torch.load(tmp_path/'retained_prefix.pt',map_location='cpu',weights_only=False)
    extended=extend_constant_progress(payload,replace(cfg,steps=8),'diffusion')
    torch.save(extended,tmp_path/'extended_progress.pt')
    resumed=train_model('diffusion',x[:80],x[80:],replace(cfg,steps=8),tmp_path/'resumed.pt',
        progress_checkpoint_path=tmp_path/'extended_progress.pt')
    direct=train_model('diffusion',x[:80],x[80:],replace(cfg,steps=8),tmp_path/'direct.pt')
    assert resumed.history==direct.history
    for a,b in zip(resumed.model.parameters(),direct.model.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key in resumed.final_model_state:
        torch.testing.assert_close(resumed.final_model_state[key],direct.final_model_state[key],rtol=0,atol=0)


import pytest


@pytest.mark.parametrize('replan',[False,True,'empirical','future_terminal','future_terminal_empirical'])
def test_terminal_cosine_ema_matches_uninterrupted_training(tmp_path, monkeypatch,replan):
    from models import training
    original_save = training._atomic_torch_save
    def retain(path, payload):
        original_save(path, payload)
        if payload.get('global_step') == 4:
            torch.save(payload, tmp_path/'prefix_state.pt')
    monkeypatch.setattr(training, '_atomic_torch_save', retain)
    x = np.random.default_rng(29).normal(size=(96, 6)).astype(np.float32)
    cfg = TrainingConfig(field_hidden_sizes=(24,16,8), time_embedding_dim=128, device='cpu',
        training_mode='fixed_steps_v1', steps=8, warmup_steps=0, validation_interval_steps=2,
        sigma_min=.01, sigma_max=64., batch_size=16, early_stopping_patience=None,
        field_preconditioning='covariance_span_v1', noise_pairing='antithetic_v1')
    prefix_config=replace(cfg,steps=12,terminal_decay_steps=4,ema_decay=.9,ema_start_step=8) if str(replan).startswith('future_terminal') else cfg
    train_model('diffusion', x[:80], x[80:], prefix_config, tmp_path/'constant.pt',
        progress_checkpoint_path=tmp_path/'constant_progress.pt')
    payload = torch.load(tmp_path/'prefix_state.pt', weights_only=True)
    terminal = replace(cfg, terminal_decay_steps=4, ema_decay=.9, ema_start_step=4)
    if replan is True:
        payload['training_config']['steps']=16
        payload['scheduler_state']['total_steps']=16
    empirical='empirical' in str(replan)
    if empirical:
        terminal=replace(terminal,training_target='empirical_posterior_v1',training_target_start_step=4)
    prepare=(prepare_empirical_target_progress if empirical else
             replan_constant_progress if replan else prepare_terminal_progress)
    torch.save(prepare(payload, terminal, 'diffusion'), tmp_path/'replay_progress.pt')
    replay = train_model('diffusion', x[:80], x[80:], terminal, tmp_path/'replay.pt',
        progress_checkpoint_path=tmp_path/'replay_progress.pt')
    direct = train_model('diffusion', x[:80], x[80:], terminal, tmp_path/'direct.pt')
    assert replay.history == direct.history
    for key in replay.model.state_dict():
        torch.testing.assert_close(replay.model.state_dict()[key], direct.model.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(replay.final_model_state[key], direct.final_model_state[key], rtol=0, atol=0)
    loaded = training.load_checkpoint(tmp_path/'replay.pt', device='cpu')
    assert loaded.history == replay.history
