
import uvicorn
import threading
import time
import os
import signal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router

import adala.web.models
from .db import engine
# f.rom adala.web.models import init_models
# from .db2 import AgentModel, SkillModel, RuntimeModel

# from peewee import SqliteDatabase

class Server:
    def __init__(self):
        
        app = FastAPI()
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )
        
        app.include_router(router)
        app.mount('/static', StaticFiles(directory='static'), name='static')

        self.app = app
        
        # db = SqliteDatabase('agents.db')
        # db.connect()

        adala.web.models.init_models(engine)
        
        # db = Database().connect()
        # db.drop_tables([ AgentModel, SkillModel, RuntimeModel, RuntimeModel.agents.get_through_model() ])
        # db.create_tables([ AgentModel, SkillModel, RuntimeModel, RuntimeModel.agents.get_through_model() ])
        
        print("Done with db")

    def start(self, args):
        global server_thread
        # self.database.connect()
        # self.database.initialize_metrics_table()
        # server_thread = threading.Thread(target=self.run_server, args=(args.port,))
        # print(server_thread)
        # server_thread.start()
        self.run_server(args.port)

    def run_server(self, port):
        uvicorn.run(self.app, host='0.0.0.0', port=port)

    def restart(self, args):
        self.shutdown(None)
        time.sleep(2)
        self.start(args)

    def shutdown(self, args):
        os.kill(os.getpid(), signal.SIGINT)
