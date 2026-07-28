import argparse
import json
import os
import time

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from . import build, evaluate, metrics, provenance, utils
from .data import RankingDataLoader
from .trainer import LOSS_CHOICES, Trainer


def train(model, trainer, epochs, metric, qrels, valid_loader, save_path, save,
          run_file, eval_every, device, config, history_file, judged_only=False,
          optimizer=None, scheduler=None, scaler=None, prov=None,
          start_epoch=0, best_valid_metric=-1.0, history=None,
          save_last=None):
    history = history or {'epoch': [], 'train_loss': [], 'val_metric': []}

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        print(f'--- epoch {epoch + 1}/{epochs}')
        train_loss = trainer.train()

        # The final epoch is always evaluated. Otherwise --epoch 5 --eval-every 2
        # never scores epoch 5, and eval_every > epochs saves no checkpoint at all.
        last_epoch = (epoch + 1) == epochs
        if (epoch + 1) % eval_every != 0 and not last_epoch:
            print(f'\tTrain Loss: {train_loss:.4f}')
            continue

        # Validation is scored in full precision regardless of the training
        # dtype: the number selects a checkpoint, so it should not be quantised.
        res_dict = evaluate.evaluate(model, valid_loader, device, amp_dtype=None)
        run_path = os.path.join(save_path, run_file)
        utils.save_trec(run_path, res_dict)

        # Fatal, not a warning. Scoring is `-c`, so topics absent from the run
        # count as unretrieved and the figure is depressed rather than
        # inflated -- but a checkpoint should not be selected on a run that is
        # quietly missing topics either way, and the validation qrels being
        # the wrong ones is the usual cause. build_data writes them per fold.
        agg = metrics.require_full_coverage(
            qrels, run_path, context='validation run',
            measures=tuple(dict.fromkeys((metric, *metrics.DEFAULT_MEASURES))),
            judged_only=judged_only)
        valid_metric = agg[metric]

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_metric'].append(valid_metric)

        # Strictly greater: a tie keeps the earlier checkpoint, so the saved
        # model is the first to reach the best score rather than the last.
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            utils.save_checkpoint(
                os.path.join(save_path, save), model, config,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch + 1, best_metric=best_valid_metric,
                provenance=prov, rng=provenance.rng_state(), history=history,
            )

        # The best checkpoint is written only when the metric improves, so it
        # is not where training got to -- resuming from it would discard every
        # epoch since and restart the optimiser trajectory from there. This one
        # is the latest state, and is what --resume should point at.
        if save_last:
            utils.save_checkpoint(
                os.path.join(save_path, save_last), model, config,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch + 1, best_metric=best_valid_metric,
                provenance=prov, rng=provenance.rng_state(), history=history,
            )

        with open(history_file, 'w') as f:
            json.dump({**history, 'best_val_metric': best_valid_metric}, f, indent=2)

        mins, secs = utils.epoch_time(start_time, time.time())
        print(f'Epoch: {epoch + 1:02} | Time: {mins}m {secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Val {metric}: {valid_metric:.4f} '
              f'| Best: {best_valid_metric:.4f}')

    return best_valid_metric


def main():
    build.handle_list_models()

    parser = argparse.ArgumentParser('Train a re-ranking baseline.')
    build.add_common_args(parser)
    parser.add_argument('--train', help='Training data.', required=True, type=str)
    parser.add_argument('--dev', help='Development data.', required=True, type=str)
    parser.add_argument('--qrels', help='Ground truth file in TREC format.',
                        required=True, type=str)
    parser.add_argument('--save-dir', help='Directory where the model is saved.',
                        required=True, type=str)
    parser.add_argument('--save', help='Checkpoint filename. Default: model.bin',
                        default='model.bin', type=str)
    parser.add_argument('--init-checkpoint', default=None, type=str,
                        help='Weights to initialise from. This does NOT resume '
                             'training: the optimiser, scheduler, scaler, epoch '
                             'counter and best metric all restart. Use --resume to '
                             'continue an interrupted run.')
    parser.add_argument('--resume', default=None, type=str,
                        help='Continue an interrupted run from a checkpoint, '
                             'restoring the optimiser, scheduler, scaler, epoch '
                             'counter, best metric and random state.')
    parser.add_argument('--allow-unverified-checkpoint', action='store_true',
                        help='Accept a checkpoint that carries no architecture '
                             'configuration. Off by default.')
    parser.add_argument('--allow-data-change', action='store_true',
                        help='Permit --resume when the input files differ from those '
                             'the checkpoint was trained on.')
    parser.add_argument('--run', help='Validation run filename. Default: dev.run',
                        default='dev.run', type=str)
    parser.add_argument('--judged-only', action='store_true',
                        help='Score with unjudged documents removed from the '
                             'ranking rather than counted as non-relevant -- '
                             'trec_eval -J. Some collections are reported that '
                             'way, and where the judgment pool is shallow the '
                             'difference is large. Off by default, matching -c.')
    parser.add_argument('--metric', default='AP', type=str,
                        help='Validation metric, as an ir_measures name: AP, '
                             'nDCG@20, P@20, RR. The trec_eval spellings (map, '
                             'ndcg_cut_20, P_20, recip_rank) are accepted as '
                             'aliases. Default: AP')
    parser.add_argument('--loss', choices=LOSS_CHOICES, default=None,
                        help="Training objective. Defaults to the one the model expects: "
                             "'cross-entropy' for cross-encoders, 'bce' for multi-vector "
                             "models. 'ce-inbatch' is a simple in-batch softmax variant "
                             "inspired by the multi-vector papers' objectives; it is not a "
                             "reproduction of either negative-sampling recipe.")
    parser.add_argument('--positives-only', action='store_true',
                        help='Drop label<=0 examples from the training file. '
                             'Required for --loss ce-inbatch.')
    parser.add_argument('--epoch', help='Number of epochs. Default: 20',
                        type=int, default=20)
    parser.add_argument('--learning-rate', help='Learning rate. Default: 1e-5.',
                        type=float, default=1e-5)
    parser.add_argument('--weight-decay', help='L2 regularisation. Default: 1e-2.',
                        type=float, default=1e-2)
    parser.add_argument('--n-warmup-steps', help='Warmup steps. Default: 10%% of total.',
                        type=int, default=None)
    parser.add_argument('--max-grad-norm', help='Gradient clipping. Default: 1.0',
                        type=float, default=1.0)
    parser.add_argument('--eval-every', help='Evaluate every N epochs. Default: 1',
                        type=int, default=1)
    parser.add_argument('--save-last', default='last.bin', type=str,
                        help='Filename for the latest-state checkpoint, written at '
                             'every evaluation and used by --resume. The main '
                             'checkpoint is the BEST model and is written only when '
                             'the metric improves, so resuming from it would discard '
                             'progress. Pass an empty string to disable and save the '
                             'disk. Default: last.bin')
    parser.add_argument('--no-data-fingerprint', action='store_true',
                        help='Skip hashing the input files. The digest is what '
                             'distinguishes the same file from a different file at the '
                             'same path; skip it only for very large inputs.')
    args = parser.parse_args()

    build.validate_common_args(parser, args)

    loss = args.loss or build.default_loss(args)
    build.validate_loss(parser, args, loss)

    if args.resume and args.init_checkpoint:
        parser.error('--resume and --init-checkpoint are different things: --resume '
                     'continues a run, --init-checkpoint starts a new one from '
                     'existing weights. Pass one.')

    for name, value in (('--epoch', args.epoch), ('--eval-every', args.eval_every)):
        if value < 1:
            parser.error(f'{name} must be >= 1')
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be > 0')
    if args.n_warmup_steps is not None and args.n_warmup_steps < 0:
        parser.error('--n-warmup-steps must be >= 0')

    pointwise = loss in ('bce', 'cross-entropy')
    if pointwise and args.positives_only:
        parser.error(
            f'--positives-only is incompatible with --loss {loss}: a pointwise '
            f'objective needs both positive and negative pairs. Filtering happens '
            f'before the label check, so this would train silently against an '
            f'all-ones target.')
    if loss == 'ce-inbatch' and not args.positives_only:
        parser.error('--loss ce-inbatch requires --positives-only: the in-batch '
                     'negatives are the other documents in the batch, so a labelled '
                     'negative would be treated as another query\'s positive.')
    if loss == 'ce-inbatch' and args.batch_size < 2:
        parser.error('--loss ce-inbatch needs a batch size of at least 2.')
    if loss == 'bce' and not args.logit_scale:
        print('NOTE  --loss bce without --logit-scale. Unnormalised inner products '
              'have std ~14, so roughly half of all pairs fall in the saturated '
              'region of the sigmoid and receive gradients near 1e-4.')

    utils.set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.cuda}'
                          if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f'Using device: {device}')

    prov = provenance.collect(
        data_files=(args.train, args.dev, args.qrels),
        digest=not args.no_data_fingerprint,
    )
    print(provenance.summarise(prov))
    if prov['git'].get('dirty'):
        print('NOTE  the working tree is modified, so the recorded commit does not '
              'fully identify the code that ran.')

    tokenizer, pretrained = build.build_tokenizer(args)
    model = build.build(args, tokenizer, pretrained)
    print(f'MODEL: {args.model} ({type(model).__name__}) | ENCODER: {pretrained} '
          f'| ENCODING: {model.ENCODING} | LOSS: {loss}')

    print('Reading train data...')
    train_set = build.make_dataset(
        args, args.train, tokenizer, train=True, model=model,
        positives_only=args.positives_only, binary_labels=pointwise,
    )
    print('Reading dev data...')
    dev_set = build.make_dataset(args, args.dev, tokenizer, train=False, model=model)

    train_loader = RankingDataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
    )
    dev_loader = RankingDataLoader(
        dataset=dev_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    arch = build.architecture_config(args, model, pretrained)

    # -- weights from an existing checkpoint -------------------------------
    ckpt = {}
    load_from = args.resume or args.init_checkpoint
    if load_from is not None:
        ckpt = utils.load_checkpoint(load_from, model, device, strict=True)
        stored_cfg = ckpt.get('config', {})
        if not stored_cfg and not args.allow_unverified_checkpoint:
            raise SystemExit(
                'The checkpoint carries no architecture configuration, so it cannot '
                'be verified. check_config skips absent keys, which means a mismatch '
                'would pass silently and every epoch after it would train the wrong '
                'model. Pass --allow-unverified-checkpoint to override.'
            )
        conflicts = utils.check_config(stored_cfg, arch, build.ARCHITECTURE_KEYS)
        if conflicts:
            for k, was, now in conflicts:
                print(f'ERROR  {k} was {was!r} in the checkpoint, {now!r} now.')
            raise SystemExit('Checkpoint was built under a different architecture. '
                             'Loading it would produce a model whose weights do not '
                             'mean what the keys say.')
        if ckpt.get('provenance'):
            print('checkpoint provenance:')
            print(provenance.summarise(ckpt['provenance']))
    model.to(device)

    if loss == 'cross-entropy':
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif loss == 'bce':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:                                       # ce-inbatch calls F.cross_entropy
        loss_fn = None

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    # Steps are optimiser steps, i.e. batches, not examples. A fixed warmup
    # count can exceed the total step count on a small fold, in which case the
    # learning rate never leaves the warmup ramp.
    total_steps = max(1, len(train_loader) * args.epoch)
    warmup = args.n_warmup_steps if args.n_warmup_steps is not None else int(0.1 * total_steps)
    if warmup >= total_steps:
        print(f'WARNING  warmup ({warmup}) >= total steps ({total_steps}); clamping to 10%.')
        warmup = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=loss_fn, scheduler=scheduler,
        metric=args.metric, data_loader=train_loader, device=device,
        loss=loss, amp_dtype=build.amp_dtype(args.amp),
        max_grad_norm=args.max_grad_norm,
    )

    # -- resume ------------------------------------------------------------
    start_epoch, best_metric, history = 0, -1.0, None
    if args.resume is not None:
        changed = provenance.compare_data(
            ckpt.get('provenance', {}).get('data', {}), prov['data'])
        if changed:
            for path, field, was, now in changed:
                print(f'ERROR  {os.path.basename(path)}: {field} was {was!r}, now {now!r}')
            if not args.allow_data_change:
                raise SystemExit(
                    'The input files differ from those the checkpoint was trained on. '
                    'Resuming would continue an optimiser trajectory fitted to '
                    'different data. Pass --allow-data-change to override.')
            print('WARNING  continuing on changed data (--allow-data-change).')

        missing = [k for k in ('optimizer_state_dict', 'scheduler_state_dict', 'epoch')
                   if k not in ckpt]
        if missing:
            raise SystemExit(
                f'--resume needs a checkpoint carrying training state; this one is '
                f'missing {missing}. It can be used with --init-checkpoint instead, '
                f'which starts a fresh run from these weights.')

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            trainer._scaler.load_state_dict(ckpt['scaler_state_dict'])
        provenance.set_rng_state(ckpt.get('rng_state', {}))
        start_epoch = int(ckpt['epoch'])
        best_metric = float(ckpt.get('best_metric', -1.0))
        history = ckpt.get('history')
        # Checked before anything is printed, so an exhausted checkpoint does
        # not first announce a start epoch beyond the limit.
        if start_epoch >= args.epoch:
            raise SystemExit(
                f'The checkpoint is already at epoch {start_epoch} and --epoch is '
                f'{args.epoch}, so there is nothing to do. Raise --epoch to continue.')
        print(f'Resuming at epoch {start_epoch + 1}/{args.epoch}, '
              f'best {args.metric} so far {best_metric:.4f}')
        if os.path.basename(args.resume) == args.save and args.save_last:
            print(f'NOTE  resuming from the BEST checkpoint. It is written only when '
                  f'the metric improves, so any epochs after {start_epoch} are being '
                  f'discarded. {args.save_last} in the same directory is the latest '
                  f'state.')

    config = {
        **arch,
        'loss': loss, 'positives_only': args.positives_only,
        'metric': args.metric, 'judged_only': args.judged_only,
        'epochs': args.epoch, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
        'warmup_steps': warmup, 'total_steps': total_steps,
        'max_grad_norm': args.max_grad_norm, 'train_amp': args.amp, 'seed': args.seed,
        'optimizer': 'AdamW',
        'validate_inputs': not args.no_validate_inputs,
        'train_file': args.train, 'dev_file': args.dev, 'qrels': args.qrels,
        'init_checkpoint': args.init_checkpoint, 'resumed_from': args.resume,
        'n_train_examples': len(train_set), 'n_dev_examples': len(dev_set),
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump({'config': config, 'provenance': prov}, f, indent=2, default=str)

    best = train(
        model=model, trainer=trainer, epochs=args.epoch, metric=args.metric,
        qrels=args.qrels, valid_loader=dev_loader, save_path=args.save_dir,
        save=args.save, run_file=args.run, eval_every=args.eval_every,
        device=device, config=config,
        history_file=os.path.join(args.save_dir, 'training_history.json'),
        judged_only=args.judged_only,
        optimizer=optimizer, scheduler=scheduler, scaler=trainer._scaler,
        prov=prov, start_epoch=start_epoch, best_valid_metric=best_metric,
        history=history, save_last=args.save_last or None,
    )
    print(f'Training complete. Best val {args.metric}: {best:.4f}')


if __name__ == '__main__':
    main()