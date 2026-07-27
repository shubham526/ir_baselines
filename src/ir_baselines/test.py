import argparse
import os

import torch
from transformers import AutoTokenizer

from . import evaluate
from . import utils
from .data.dataloader import RankingDataLoader
from .data.dataset import RankingDataset
from .models import ENCODER_MAP, add_model_args, build_model


def main():
    parser = argparse.ArgumentParser('Run inference with a fine-tuned baseline.')
    add_model_args(parser)
    parser.add_argument('--test', help='Test data.', required=True, type=str)
    parser.add_argument('--checkpoint', help='Checkpoint to load.', required=True, type=str)
    parser.add_argument('--save-dir', help='Directory where the run is written.',
                        required=True, type=str)
    parser.add_argument('--run', help='Output run file in TREC format.', required=True, type=str)
    parser.add_argument('--run-tag', help='Run tag written in field 6. Default: the model name.',
                        default=None, type=str)
    parser.add_argument('--max-len', help='Max length for pair encoding. Default: 512',
                        default=512, type=int)
    parser.add_argument('--max-query-len', help='Max query length for dual encoding. Default: 20',
                        default=20, type=int)
    parser.add_argument('--max-doc-len', help='Max document length for dual encoding. Default: 512',
                        default=512, type=int)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--num-workers', help='DataLoader workers. Default: 8', type=int, default=8)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))
    device = torch.device(cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu')

    vocab = ENCODER_MAP[args.pretrain]
    print('MODEL: {} | ENCODER: {}'.format(args.model, args.pretrain))
    tokenizer = AutoTokenizer.from_pretrained(vocab)

    model = build_model(args)

    print('Loading checkpoint...')
    stored = utils.load_checkpoint(args.checkpoint, model, device)
    print('[Done].')

    # A T5 checkpoint trained with one pooling and evaluated with the other
    # loads with every key matched and produces different scores throughout,
    # so the stored setting is checked rather than trusted.
    if stored.get('t5_pooling') and stored['t5_pooling'] != args.t5_pooling:
        raise SystemExit(
            'ERROR  this checkpoint was trained with --t5-pooling {!r}, and '
            '{!r} was requested. The weights load either way and the scores '
            'differ throughout, so this is refused.'.format(
                stored['t5_pooling'], args.t5_pooling))

    print('Reading test data...')
    test_set = RankingDataset(
        dataset=args.test, tokenizer=tokenizer, train=False,
        encoding=model.ENCODING, max_len=args.max_len,
        max_query_len=args.max_query_len, max_doc_len=args.max_doc_len)
    print('[Done].')

    print('Creating data loader...')
    test_loader = RankingDataLoader(dataset=test_set, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)
    print('[Done].')

    print('Using device: {}'.format(device))
    model.to(device)

    print('Running inference...')
    res_dict = evaluate.evaluate(model=model, data_loader=test_loader, device=device)

    os.makedirs(args.save_dir, exist_ok=True)
    run_path = os.path.join(args.save_dir, args.run)
    utils.save_trec(run_path, res_dict, run_tag=args.run_tag or args.model)

    # A run that silently loses pairs still scores without error and reports a
    # plausible figure, so the counts are compared here.
    n_in = len(test_set)
    n_out = sum(len(v) for v in res_dict.values())
    print('Examples in test file : {}'.format(n_in))
    print('Pairs written to run  : {} across {} queries'.format(n_out, len(res_dict)))
    if n_in != n_out:
        print('WARNING: {} example(s) did not reach the run file.'.format(n_in - n_out))

    print('Run file saved to ==> {}'.format(run_path))


if __name__ == '__main__':
    main()