
import argparse
from flask import Flask
import threading
import os
import signal
import time
import sqlite3

import argparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
import os
import sqlite3


import argparse
from server import Server, app
from data_processing import DataProcessing

    
parser = argparse.ArgumentParser(description="ADALA Command Line Interface")
subparsers = parser.add_subparsers(dest='command')
server = Server()
data_processing = DataProcessing()
utility = Utility()

# Server
server_parser = subparsers.add_parser('server')
server_parser.add_argument('--port', type=int, default=8000)
server_parser.set_defaults(func=server.start)

server_instance = Server(app, port=args.port)
server_instance.run()

# Restart
restart_parser = subparsers.add_parser('restart')
restart_parser.add_argument('--port', type=int, default=8000)
restart_parser.set_defaults(func=server.restart)

# Shutdown
shutdown_parser = subparsers.add_parser('shutdown')
shutdown_parser.set_defaults(func=server.shutdown)

# Send data
send_parser = subparsers.add_parser('send')
send_parser.add_argument('--file', required=True, help='Path to data file')
send_parser.set_defaults(func=data_processing.send)

# Predict
predict_parser = subparsers.add_parser('predict')
predict_parser.add_argument('--file', required=True, help='Path to test data file')
predict_parser.add_argument('--output', required=True, help='Path to save the output')
predict_parser.set_defaults(func=data_processing.predict)

# List models
list_models_parser = subparsers.add_parser('list-models')
list_models_parser.set_defaults(func=utility.list_models)

# List agents
list_agents_parser = subparsers.add_parser('list-agents')
list_agents_parser.set_defaults(func=utility.list_agents)

# Logs
logs_parser = subparsers.add_parser('logs')
logs_parser.add_argument('--tail', type=int, default=10)
logs_parser.set_defaults(func=utility.logs)

# Metrics
metrics_parser = subparsers.add_parser('metrics')
metrics_parser.set_defaults(func=utility.metrics)

# Help
help_parser = subparsers.add_parser('help')
help_parser.add_argument('command', type=str, help='Command to get help for')
help_parser.set_defaults(func=utility.help)


if __name__ == "__main__":
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
