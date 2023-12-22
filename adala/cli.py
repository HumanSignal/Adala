import argparse
import sys
from rich import print

from adala.utils.gsheet import read_gsheet, write_gsheet
from adala.agents.base import create_agent_from_file


def run_on_gsheet(config_file, input_file, worksheet):
    print(f'Creating agent from {config_file}...')
    agent = create_agent_from_file(config_file)
    print(agent)
    print(f'Reading input from {input_file}...')
    input = read_gsheet(input_file, worksheet)
    print(f'Running agent on {input.shape[0]} rows...')
    predictions = agent.run(input)
    print(f'Writing predictions to {input_file} ({worksheet})...')
    write_gsheet(predictions, input_file, worksheet)
    print('Done!')


def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Welcome to Adala!")
    parser.add_argument("--config-file", type=str, help="Path to the config file (agent.yml)")
    parser.add_argument("--input-file", type=str, help="Path to the input file")
    parser.add_argument("--worksheet", type=str, help="Worksheet name")
    return parser.parse_args(args)


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:])
    run_on_gsheet(arguments.config_file, arguments.input_file, arguments.worksheet)