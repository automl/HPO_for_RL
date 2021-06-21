import os
import logging
import Pyro4

from pbt.network import Daemon, WorkerAdapter
from pbt.tqdm_logger import TqdmLoggingHandler
CONTROLLER_URI_FILENAME = 'controller_uri.txt'


@Pyro4.expose
class ControllerDaemon(Daemon):
    def __init__(self, controller):
        self.logger = logging.getLogger('pbt')
        self.logger.setLevel(logging.DEBUG)
        # self.logger.addHandler(TqdmLoggingHandler())
        self.controller = controller
        self.pyro_daemon = None

    def start(self):
        Pyro4.config.SERVERTYPE = 'multiplex'
        self.pyro_daemon = Pyro4.Daemon(host=self._get_hostname())
        uri = self.pyro_daemon.register(self)
        self._save_pyro_uri(uri)
        self.pyro_daemon.requestLoop()

    def register_worker_by_uri(self, uri):
        self.controller.register_worker(WorkerAdapter(Pyro4.Proxy(uri)))

    def request_trial(self):
        return self.controller.request_trial()

    def send_evaluation(self, member_id, score):
        self.controller.send_evaluation(member_id, score)

    def shut_down(self):
        self.pyro_daemon.shutdown()

    def _save_pyro_uri(self, uri):
        save_path = os.path.join(
            self.controller.data_path, CONTROLLER_URI_FILENAME)
        if not os.path.isdir(self.controller.data_path):
            os.makedirs(self.controller.data_path)
        with open(save_path, 'w') as f:
            f.write(str(uri))
        self.logger.info(f'Saved pyro uri at {save_path}.')
