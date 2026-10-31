"""Reproduce the frozen design in a new directory without overwriting results.

Use a separately authorized budget for a new solver campaign. The provided
verify_submission command reproduces statistics without new solver executions.
"""
import argparse
from pathlib import Path
import final_study as study


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('stage',choices=['freeze','run','timing-plan','repair','analyze'])
    args=parser.parse_args();target=Path(args.output).resolve()
    if target==study.ROOT.resolve():parser.error('Choose a new output directory')
    if args.stage=='freeze' and target.exists():parser.error('Freeze requires a new output directory')
    study.ROOT=target
    {'freeze':study.freeze,'run':study.run,'timing-plan':study.timing_plan,'repair':study.repair,'analyze':study.analyze}[args.stage]()


if __name__=='__main__':main()
