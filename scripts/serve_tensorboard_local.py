"""Serve TensorBoard with a bounded startup port probe on WSL loopback."""
import socket
import sys

from tensorboard import program


def local_server(app, flags):
    # The installed TensorBoard probes localhost with connect_ex before bind.
    # WSL can silently drop that probe on an unused port, causing a long stall.
    previous = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(1.)
        server = program.WerkzeugServer(app, flags)
        server.socket.settimeout(None)
        return server
    finally:
        socket.setdefaulttimeout(previous)


if __name__ == '__main__':
    board = program.TensorBoard(server_class=local_server)
    board.configure(argv=['tensorboard', *sys.argv[1:]])
    raise SystemExit(board.main())
