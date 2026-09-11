"""Explicit diagnostic export of native-best weights from a retained progress state.

This never marks the parent training complete. Its receipt records the actual
completed prefix and the originally planned budget. The exported copy follows
the source portable schema and is checked by its strict loader.
"""
import argparse,json,hashlib
from pathlib import Path
import torch
from models.training import FIXED_STEP_CHECKPOINT_SCHEMA_VERSION,_fixed_weights_metadata,load_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    a=p.parse_args();source=a.run/f'progress_{a.step:06d}.pt'
    v=torch.load(source,map_location='cpu',weights_only=False);assert v['global_step']==a.step
    destination=a.run/f'interim_{a.step:06d}';destination.mkdir(exist_ok=False)
    cfg={**v['training_config'],'steps':a.step};batch=cfg['batch_size']
    weights=_fixed_weights_metadata(best_state=v['best_state'],final_state=v['model_state'],best_step=v['best_step'],
        final_step=a.step,batch_size=batch,initial_validation_loss=v['initial_validation_loss'])
    out=dict(schema_version=FIXED_STEP_CHECKPOINT_SCHEMA_VERSION,family=v['family'],model_contract=v['model_contract'],
        architecture=v['architecture'],training_config=cfg,model_state=v['best_state'],final_model_state=v['model_state'],
        history=v['history'],best_step=v['best_step'],best_validation_loss=v['best_validation_loss'],global_step=a.step,
        examples_seen_total=a.step*batch,examples_seen_at_selected_checkpoint=v['best_step']*batch,
        weights_metadata=weights,normalization=v['normalization'])
    torch.save(out,destination/'model.pt');loaded=load_checkpoint(destination/'model.pt',device='cpu')
    manifest=json.loads((a.run/'manifest.json').read_text())
    manifest.update(interim=True,planned_steps=v['training_config']['steps'],completed_steps=a.step,
        source_progress_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config=cfg,parent_run=a.run.name)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (destination/'interim.json').write_text(json.dumps(dict(status='interim',completed_steps=a.step,
        planned_steps=v['training_config']['steps'],checkpoint_sha256=loaded.checkpoint_sha256))+'\n')
    print(destination,flush=True)


if __name__=='__main__':main()
