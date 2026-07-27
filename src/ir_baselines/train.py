import argparse
import json
import os
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from . import evaluate
from . import metrics
from . import utils
from .data.dataloader import RankingDataLoader
from .data.dataset import RankingDataset
from .models import ENCODER_MAP, add_model_args, build_model
from .trainer import Trainer


def train(model, trainer, epochs, metric, qrels, valid_loader, save_path, save,
          run_file, eval_every, device, config):
    best_valid_metric = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        print('Training on train set...')
        train_loss = trainer.train()

        if (epoch + 1) % eval_every == 0:
            print('Evaluating on dev set...')
            res_dict = evaluate.evaluate(model=model, data_loader=valid_loader, device=device)

            utils.save_trec(os.path.join(save_path, run_file), res_dict)
            valid_metric = metrics.get_metric(qrels, os.path.join(save_path, run_file), metric)

            if valid_metric >= best_valid_metric:
                best_valid_metric = valid_metric
                utils.save_checkpoint(os.path.join(save_path, save), model, config)

            epoch_mins, epoch_secs = utils.epoch_time(start_time, time.time())
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train Loss: {train_loss:.3f}| Val. Metric: {valid_metric:.4f} '
                  f'| Best Val. Metric: {best_valid_metric:.4f}')


def main():
    parser = argparse.ArgumentParser('Train a re-ranking baseline.')
    add_model_args(parser)
    parser.add_argument('--train', help='Training data.', required=True, type=str)
    parser.add_argument('--dev', help='Development data.', required=True, type=str)
    parser.add_argument('--qrels', help='Ground truth file in TREC format.', required=True, type=str)
    parser.add_argument('--save-dir', help='Directory where the model is saved.', required=True, type=str)
    parser.add_argument('--save', help='Name of checkpoint to save. Default: model.bin',
                        default='model.bin', type=str)
    parser.add_argument('--checkpoint', help='Checkpoint to initialise from. Default: None',
                        default=None, type=str)
    parser.add_argument('--run', help='Validation run file. Default: dev.run', default='dev.run', type=str)
    parser.add_argument('--metric', help='Validation metric. Default: map', default='map', type=str)
    parser.add_argument('--max-len', help='Max length for pair encoding. Default: 512', default=512, type=int)
    parser.add_argument('--max-query-len', help='Max query length for dual encoding. Default: 20',
                        default=20, type=int)
    parser.add_argument('--max-doc-len', help='Max document length for dual encoding. Default: 512',
                        default=512, type=int)
    parser.add_argument('--epoch', help='Number of epochs. Default: 20', type=int, default=20)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--learning-rate', help='Learning rate. Default: 2e-5.', type=float, default=2e-5)
    parser.add_argument('--n-warmup-steps', help='Warmup steps. Default: 1000.', type=int, default=1000)
    parser.add_argument('--eval-every', help='Evaluate every N epochs. Default: 1', type=int, default=1)
    parser.add_argument('--num-workers', help='DataLoader workers. Default: 8', type=int, default=8)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))
    device = torch.device(cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu')

    pretrain = vocab = ENCODER_MAP[args.pretrain]
    print('MODEL: {} | ENCODER: {}'.format(args.model, args.pretrain))

    model = build_model(args)

    # Written before training so that an interrupted run still records what it
    # was doing. The directory is created here: earlier versions wrote
    # config.json into a directory that might not exist and crashed.
    os.makedirs(args.save_dir, exist_ok=True)
    config = {
        'model': args.model,
        'Max Input': args.max_len,
        'Model': pretrain,
        'Metric': args.metric,
        'Epochs': args.epoch,
        'Batch Size': args.batch_size,
        'Learning Rate': args.learning_rate,
        'Warmup Steps': args.n_warmup_steps,
        **model.config_dict(),
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        f.write('%s\n' % json.dumps(config))

    tokenizer = AutoTokenizer.from_pretrained(vocab)

    print('Reading train data...')
    train_set = RankingDataset(
        dataset=args.train, tokenizer=tokenizer, train=True,
        encoding=model.ENCODING, max_len=args.max_len,
        max_query_len=args.max_query_len, max_doc_len=args.max_doc_len)
    print('[Done].')

    print('Reading dev data...')
    dev_set = RankingDataset(
        dataset=args.dev, tokenizer=tokenizer, train=False,
        encoding=model.ENCODING, max_len=args.max_len,
        max_query_len=args.max_query_len, max_doc_len=args.max_doc_len)
    print('[Done].')

    print('Creating data loaders...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))
    train_loader = RankingDataLoader(dataset=train_set, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)
    dev_loader = RankingDataLoader(dataset=dev_set, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
    print('[Done].')

    loss_fn = nn.CrossEntropyLoss() if model.LOSS == 'cross_entropy' else nn.BCEWithLogitsLoss()

    if args.checkpoint is not None:
        print('Loading checkpoint...')
        # Through utils, so that training and inference agree about the file
        # layout. Earlier versions loaded inline here with no map_location and
        # no key remapping, and failed on checkpoints test.py could read.
        utils.load_checkpoint(args.checkpoint, model, device)
        print('[Done].')

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.n_warmup_steps,
        num_training_steps=len(train_set) * args.epoch // args.batch_size)

    print('Using device: {}'.format(device))
    model.to(device)
    loss_fn.to(device)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=loss_fn,
                      scheduler=scheduler, metric=args.metric,
                      data_loader=train_loader, device=device)

    train(model=model, trainer=trainer, epochs=args.epoch, metric=args.metric,
          qrels=args.qrels, valid_loader=dev_loader, save_path=args.save_dir,
          save=args.save, run_file=args.run, eval_every=args.eval_every,
          device=device, config=config)

    print('Training complete.')


if __name__ == '__main__':
    main()