import argparse
import json
import os

import torch

from . import build, evaluate, metrics, provenance, utils
from .data import RankingDataLoader


def main():
    build.handle_list_models()

    parser = argparse.ArgumentParser('Run inference with a fine-tuned baseline.')
    build.add_common_args(parser)
    parser.add_argument('--test', help='Test data.', required=True, type=str)
    parser.add_argument('--save-dir', help='Directory where the run is written.',
                        required=True, type=str)
    parser.add_argument('--run', help='Output run filename.', required=True, type=str)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load. Required unless --allow-untrained.')
    parser.add_argument('--allow-untrained', action='store_true',
                        help='Run without a checkpoint. For debugging only; the '
                             'resulting run is not a model output.')
    parser.add_argument('--allow-unverified-checkpoint', action='store_true',
                        help='Accept a checkpoint that carries no architecture '
                             'configuration. Off by default, because several '
                             'mismatches cannot be detected from state-dict keys alone.')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Run tag written in field 6. Default: the model name.')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Truncate each ranking to K documents. Default: no truncation.')
    parser.add_argument('--qrels', type=str, default=None,
                        help='Optional qrels; if given, the run is scored and its topic '
                             'set is checked against the qrels.')
    parser.add_argument('--expected-topics', type=int, default=None,
                        help='Fail if the written run has a different number of topics. '
                             'Use when concatenating folds.')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Permit a run whose topic set differs from the qrels. '
                             'Off by default.')
    parser.add_argument('--no-run-provenance', action='store_true',
                        help='Skip writing <run>.provenance.json. A run file cannot '
                             'carry its own provenance -- every TREC parser rejects a '
                             'comment line -- so this is the only record of what '
                             'produced it.')
    parser.add_argument('--tag-commit', action='store_true',
                        help='Append the short commit hash to the run tag, e.g. '
                             '"rankt5.a3f9c21", so a run file separated from its '
                             'sibling still names the code that made it. Skipped when '
                             'the working tree is dirty, since the hash would not '
                             'identify what ran.')
    args = parser.parse_args()

    build.validate_common_args(parser, args)
    if args.top_k is not None and args.top_k < 1:
        parser.error('--top-k must be >= 1')
    if args.checkpoint is None and not args.allow_untrained:
        parser.error('--checkpoint is required. An untrained model produces a run file '
                     'that is indistinguishable from a real one at a glance; pass '
                     '--allow-untrained if that is genuinely what you want.')
    if args.amp != 'none':
        print('NOTE  --amp affects training only; inference runs in full precision.')

    utils.set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.cuda}'
                          if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f'Using device: {device}')

    tokenizer, pretrained = build.build_tokenizer(args)
    model = build.build(args, tokenizer, pretrained)
    print(f'MODEL: {args.model} ({type(model).__name__}) | ENCODER: {pretrained} '
          f'| ENCODING: {model.ENCODING}')

    print('Reading test data...')
    test_set = build.make_dataset(args, args.test, tokenizer, train=False, model=model)
    test_loader = RankingDataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    arch = build.architecture_config(args, model, pretrained)
    stored, ckpt_prov = {}, None

    if args.checkpoint is not None:
        # strict=True catches renamed or missing parameters. It does NOT catch
        # every mismatch: with shared_encoder=True the two encoders are one
        # module, so a separate-tower checkpoint loads with every key matched and
        # the document tower silently overwrites the query tower. The same is
        # true of the T5 pooling setting, which is not a parameter at all. The
        # config comparison below is the only thing that catches either.
        ckpt = utils.load_checkpoint(args.checkpoint, model, device, strict=True)
        stored = ckpt.get('config', {})
        if not stored:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                    'config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    sidecar = json.load(f)
                # config.json holds {'config': ..., 'provenance': ...}; older
                # ones held the config at the top level.
                stored = sidecar.get('config', sidecar)
                print(f'Using sibling {cfg_path} for provenance.')

        # What produced these weights. Printed rather than checked: a different
        # commit or environment is not an error, but it is the first thing worth
        # knowing when a number does not match.
        ckpt_prov = ckpt.get('provenance')
        if ckpt_prov:
            print(provenance.summarise(ckpt_prov))
        elif stored:
            print('NOTE  checkpoint carries configuration but no provenance, so the '
                  'code and environment that produced it are not recorded.')

        conflicts = utils.check_config(stored, arch, build.ARCHITECTURE_KEYS)
        if conflicts:
            for k, was, now in conflicts:
                print(f'ERROR  {k} was {was!r} at training time, {now!r} now.')
            raise SystemExit(
                'Inference settings differ from training settings. The weights do '
                'not mean what this architecture expects, and load_state_dict will '
                'not tell you so. Pass the matching flags.'
            )
        if not stored:
            if not args.allow_unverified_checkpoint:
                raise SystemExit(
                    'This checkpoint carries no architecture configuration, so the '
                    'inference settings cannot be verified against the ones that '
                    'produced the weights. Several mismatches load with every key '
                    'matched and no warning, so refusing rather than guessing. '
                    'Retrain to embed the config, or pass '
                    '--allow-unverified-checkpoint.'
                )
            print('WARNING  checkpoint carries no configuration; settings unverified.')
    else:
        ckpt_prov = None
        print('WARNING  running an untrained model (--allow-untrained).')

    model.to(device)

    print('Running inference...')
    res_dict = evaluate.evaluate(model, test_loader, device, amp_dtype=None)

    run_tag = args.run_tag or args.model
    if args.tag_commit:
        commit = provenance.short_commit()
        if commit:
            run_tag = f'{run_tag}.{commit}'
        else:
            print('NOTE  --tag-commit had no clean commit to use (not a repository, '
                  'or the tree is modified); the tag is unchanged.')

    run_path = os.path.join(args.save_dir, args.run)
    utils.save_trec(run_path, res_dict, run_tag=run_tag, top_k=args.top_k)

    if not args.no_run_provenance:
        prov_path = provenance.write_run_provenance(
            run_path,
            inference=provenance.collect(
                data_files=(args.test, args.qrels),
                digest=True,
            ),
            checkpoint_path=args.checkpoint,
            checkpoint_config=stored or None,
            checkpoint_provenance=ckpt_prov,
        )
        print(f'Provenance written to {prov_path}')

    run_topics = utils.run_topic_set(run_path)
    n_written = sum(len(v) for v in res_dict.values())
    print(f'Run written to {run_path}')
    print(f'  examples in test file : {len(test_set)}')
    print(f'  pairs written         : {n_written} across {len(run_topics)} topics')
    if args.top_k is None and n_written != len(test_set):
        print(f'  WARNING  {len(test_set) - n_written} example(s) did not reach the run '
              f'file. A run that silently loses pairs still scores without error and '
              f'reports a plausible figure.')

    if args.expected_topics is not None and len(run_topics) != args.expected_topics:
        raise SystemExit(
            f'ERROR  run has {len(run_topics)} topics, expected {args.expected_topics}. '
            f'A fold-concatenated run that is short on topics still evaluates '
            f'without error and reports a plausible number, so this is checked '
            f'here rather than left to be discovered later.'
        )

    if args.qrels:
        qrels_topics = utils.qrels_topic_set(args.qrels)
        missing, extra = qrels_topics - run_topics, run_topics - qrels_topics
        if (missing or extra) and not args.allow_partial:
            raise SystemExit(
                f'ERROR  topic sets differ: {len(missing)} qrels topics absent from '
                f'the run (e.g. {sorted(missing)[:5]}), {len(extra)} run topics absent '
                f'from the qrels (e.g. {sorted(extra)[:5]}). The right count with the '
                f'wrong ids passes a count check and fails this one. Pass '
                f'--allow-partial to score anyway.'
            )
        agg, scored, qrels_topics, _ = metrics.get_all_metrics(args.qrels, run_path)
        print(f'Scored {len(scored)} of {len(qrels_topics)} qrels topics.')
        if scored != qrels_topics:
            unscored = qrels_topics - scored
            print(f'NOTE  {len(unscored)} qrels topics are absent from the run '
                  f'(e.g. {sorted(unscored)[:5]}) and count as unretrieved, '
                  f'which is what trec_eval -c does.')
        for m in metrics.DEFAULT_MEASURES:
            if m in agg:
                print(f'  {m:<10} {agg[m]:.4f}')


if __name__ == '__main__':
    main()