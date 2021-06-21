from threading import Thread

import Pyro4

from pbt.network import Daemon


@Pyro4.expose
class WorkerDaemon(Daemon):
    def __init__(self, worker):
        self.worker = worker
        self.pyro_daemon = None

    @property
    def worker_id(self):
        return self.worker.worker_id

    def start(self):
        self.pyro_daemon = Pyro4.Daemon(host=self._get_hostname())
        uri = self.pyro_daemon.register(self)
        thread = Thread(target=self.pyro_daemon.requestLoop)
        thread.start()
        return uri

    def ping(self):
        return True

    def stop(self):
        self.worker.stop()
        self.pyro_daemon.shutdown()
