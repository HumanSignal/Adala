import argparse
import sys
import pandas as pd
from rich import print
from adala.utils.gsheet import read_gsheet, write_gsheet
from adala.agents.base import create_agent_from_file


def run_gsheet(agent, input_file, worksheet):
    print(f"Reading input from {input_file}...")
    input = read_gsheet(input_file, worksheet)
    print(f"Running agent on {input.shape[0]} rows...")
    predictions = agent.run(input)
    print(f"Writing predictions to {input_file} ({worksheet})...")
    write_gsheet(predictions, input_file, worksheet)


def _run_df(agent, df, output_file):
    print(f"Running agent on {df.shape[0]} rows...")
    predictions = agent.run(df)
    if output_file is None:
        print("Result:")
        print(predictions)
    else:
        predictions.to_csv(output_file, index=False)


def run_csv(agent, input_file, output_file):
    print(f"Reading input from {input_file}...")
    input = pd.read_csv(input_file)
    _run_df(agent, input, output_file)


def run_dataset(agent, input_file, dataset_config, dataset_split, output_file):
    from datasets import load_dataset

    ds = load_dataset(input_file, dataset_config, split=dataset_split)
    input = pd.DataFrame(ds)
    _run_df(agent, input, output_file)


def run(args):
    print(f"Creating agent from {args.workflow}...")
    agent = create_agent_from_file(args.workflow)
    print(agent)
    if args.file_type == "gsheet":
        run_gsheet(agent, args.input, args.worksheet)
    elif args.file_type == "csv":
        run_csv(agent, args.input, args.output)
    elif args.file_type == "dataset":
        run_dataset(
            agent, args.input, args.dataset_config, args.dataset_split, args.output
        )
    else:
        raise ValueError(f"Unknown file type {args.file_type}")
    print("Done!")


def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Welcome to Adala!")
    parser.add_argument(
        "--workflow", type=str, help="Path to the workflow file (workflow.yml)"
    )
    parser.add_argument("--input", type=str, help="Path to the input file")
    parser.add_argument(
        "--file-type",
        type=str,
        choices=["csv", "gsheet", "dataset"],
        help="File type",
        default="csv",
    )
    parser.add_argument(
        "--output", type=str, help="Output file", required=False, default=None
    )
    parser.add_argument("--worksheet", type=str, help="Worksheet name", required=False)
    parser.add_argument(
        "--dataset-config", type=str, help="Dataset config name", required=False
    )
    parser.add_argument(
        "--dataset-split", type=str, help="Dataset split name", required=False
    )
    return parser.parse_args(args)


def main():
    args = parse_arguments(sys.argv[1:])
    run(args)


if __name__ == "__main__":
    main()
