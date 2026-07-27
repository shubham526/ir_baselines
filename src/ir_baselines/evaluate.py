import torch
import tqdm


def evaluate(model, data_loader, device):
    """
    Score every pair in the loader.

    The score transform belongs to the model: a two-way classifier writes the
    softmax probability of the relevant class, a single-score model writes its
    score. evaluate() only calls model.score().
    """
    rst_dict = {}
    model.eval()

    with torch.no_grad():
        for dev_batch in tqdm.tqdm(data_loader, total=len(data_loader)):
            query_id, doc_id, label = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label']
            batch_score = model.score(dev_batch, device)
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]

    return rst_dict
