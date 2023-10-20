
class Server:
    def start(self, args):
        global server_thread
        self.database.connect()
        self.database.initialize_metrics_table()
        server_thread = threading.Thread(target=self.run_server, args=(args.port,))
        server_thread.start()

    def run_server(self, port):
        app.run(host='0.0.0.0', port=port)

    def restart(self, args):
        self.shutdown(None)
        time.sleep(2)
        self.start(args)

    def shutdown(self, args):
        os.kill(os.getpid(), signal.SIGINT)
