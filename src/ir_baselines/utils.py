import torch


def save_trec(rst_file, rst_dict, run_tag='BASELINE'):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id + ' Q0 ' + str(value[0]) + ' ' + str(rank + 1) + ' '
                             + str(value[1][0]) + ' ' + run_tag + '\n')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(save_path, model, config=None):
    """
    Save weights together with the configuration that produced them.

    The configuration travels inside the checkpoint so that a file which is
    copied or renamed keeps its provenance. test.py checks it: a T5 checkpoint
    trained with one pooling and evaluated with the other loads with every key
    matched and silently produces different scores.
    """
    if save_path is None:
        return
    torch.save({'model_state_dict': model.state_dict(), 'config': config or {}}, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    """
    Load a checkpoint and return the configuration stored in it.

    Three layouts are accepted, because all three exist among the released
    checkpoints:

      - {'model_state_dict': ..., 'config': ...}, written by save_checkpoint
      - a bare state_dict, written by earlier versions of save_checkpoint
      - keys prefixed 't5.' rather than 'encoder.', from runs produced before
        the attribute was renamed. Same tensors, same shapes.
    """
    if load_path is None:
        return {}
    obj = torch.load(load_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        state_dict, config = obj['model_state_dict'], obj.get('config', {})
    else:
        state_dict, config = obj, {}
        print('NOTE  checkpoint carries no config; settings cannot be verified.')
    if any(k.startswith('t5.') for k in state_dict):
        state_dict = {('encoder.' + k[3:] if k.startswith('t5.') else k): v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f'Model loaded from <== {load_path}')
    return config
