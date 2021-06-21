import socket


class Daemon:
    def _get_hostname(self):
        my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        my_socket.connect(('10.255.255.255', 1))
        ip = my_socket.getsockname()[0]
        my_socket.close()
        return ip
