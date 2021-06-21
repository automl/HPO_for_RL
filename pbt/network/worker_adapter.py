from Pyro4.errors import CommunicationError


class WorkerAdapter:
    def __init__(self, worker):
        self._worker = worker

    @property
    def worker_id(self):
        return self._worker.worker_id

    def ping(self):
        try:
            return self._worker.ping()
        except CommunicationError:
            return False

    def stop(self):
        try:
            self._worker.stop()
        except CommunicationError:
            pass
