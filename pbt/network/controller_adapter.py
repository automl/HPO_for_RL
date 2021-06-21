from Pyro4.errors import ConnectionClosedError, CommunicationError

from pbt.population import NoTrial


class ControllerAdapter:
    def __init__(self, controller):
        self._controller = controller

    def register_worker_by_uri(self, uri):
        try:
            self._controller.register_worker_by_uri(uri)
            return True
        except CommunicationError:
            # URI from file is not valid -> kill worker
            return False

    def request_trial(self):
        try:
            return self._controller.request_trial()
        except ConnectionClosedError:
            return NoTrial().to_tuple()

    def send_evaluation(self, member_id, score):
        try:
            self._controller.send_evaluation(member_id, float(score))
        except ConnectionClosedError:
            return
