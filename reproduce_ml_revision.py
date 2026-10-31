"""Run a revision stage in an explicit new output directory, preserving evidence.

Example (run through compute_budget for experiments):
python -m reproduce_ml_revision --output results/recovery/reproduction freeze
"""
import argparse
from pathlib import Path
from mapf import ml_revision as campaign


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('stage',choices=['freeze','train-a','audit','bulk','train-b','evaluate','summarize'])
    args=parser.parse_args();target=Path(args.output).resolve()
    if target==campaign.ROOT.resolve():
        parser.error('Choose a new directory; delivered evidence is read-only through this entry point')
    if args.stage=='freeze' and target.exists():
        parser.error('Freeze requires a new output directory')
    campaign.ROOT=target
    if args.stage=='freeze':campaign.freeze()
    else:
        stages={'train-a':campaign.train_a,'audit':campaign.audit_stage,'bulk':campaign.bulk,
                'train-b':campaign.train_b,'evaluate':campaign.evaluate,'summarize':campaign.summarize}
        stages[args.stage](campaign.protocol())


if __name__=='__main__':main()
