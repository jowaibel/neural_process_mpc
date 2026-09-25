"""TCP server wrapping QubeSimulator for the C++ MPC clients
(src_cpp/tests/run_furuta_np_casadi_client.cpp, run_furuta_np_laopt_client.cpp,
run_furuta_eq_laopt_client.cpp).

The simulator itself (QubeSimulator's own background process) always keeps
running in real time, integrating at `--sim-rate` (default: its own
QubeBase.frequency, 4000 Hz) and
applying whatever torque setpoint was last set -- with no client connected,
that's just the last (or default zero) value, so it "emulates the real
world" regardless of whether an MPC client is attached.

On top of that, this server accepts a single client connection (one
experiment per server run: when that client disconnects, the server saves
the log and shuts down) and, for its duration: pushes the current state at a fixed cadence of `--state-rate`
(default 200 Hz, independent of the integration rate and of the MPC model's
`dt`) as length-prefixed JSON `{"t": <time>, "x": [...]}`, and applies
whatever `{"u": [...]}` messages it receives to the simulator's torque
setpoint as soon as they arrive. An input message may also carry the
client's MPC computation time, `{"u": [...], "solve_ms": <ms>}`, which is
logged (see --dump).

Wire format: each message is a 4-byte big-endian length prefix followed by
that many bytes of UTF-8 JSON.

If `--dump` is given (or, absent that, if the MPC config's `save` is true),
the session's closed-loop log (t, x, u -- one row per state sample sent,
tagged with whichever torque command was most recently applied; plus
solve_t, solve_ms -- one entry per input message that reported a solve time) is
written to `<dump>.npz` under model/ when the client disconnects,
overwriting any previous dump of that name. `--dump` defaults to the MPC
config's `save_path`. After shutting down, the server opens the saved dump
with scripts/plot_cpp_closed_loop.py (unless --no-plot).
"""
import argparse
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from npmpc.hardware import create_hardware


def send_msg(conn, obj: dict) -> None:
    payload = json.dumps(obj).encode('utf-8')
    conn.sendall(struct.pack('>I', len(payload)) + payload)


def recv_exact(conn, n: int):
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(conn):
    header = recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack('>I', header)
    payload = recv_exact(conn, length)
    if payload is None:
        return None
    return json.loads(payload.decode('utf-8'))


class ClosedLoopLog:
    """Shared, single-writer-per-field log: sender_loop appends state rows,
    receiver_loop only updates `last_u` and appends the client's solve times
    (single assignments/appends are atomic under the GIL, so no lock is
    needed between the two threads)."""

    def __init__(self):
        self.last_u = 0.0
        self.t = []
        self.x = []
        self.u = []
        self.solve_t = []   # arrival time of each input message that carried a solve time
        self.solve_ms = []  # MPC computation time reported by the client [ms]

    def record(self, t: float, x: list) -> None:
        self.t.append(t)
        self.x.append(x)
        self.u.append(self.last_u)

    def record_solve(self, t: float, solve_ms: float) -> None:
        self.solve_t.append(t)
        self.solve_ms.append(solve_ms)

    def save(self, path: Path, dt: float, state_period: float, sim_period: float) -> bool:
        """Writes the log to `path`; returns False (and writes nothing) if it is empty."""
        if not self.t:
            return False
        np.savez(path,
                  t=np.array(self.t), x=np.array(self.x), u=np.array(self.u),
                  solve_t=np.array(self.solve_t), solve_ms=np.array(self.solve_ms),
                  dt=dt, state_period=state_period, sim_period=sim_period)
        print(f'Saved closed-loop log to {path} ({len(self.t)} steps)')
        return True


def plot_log(path: Path) -> None:
    """Opens scripts/plot_cpp_closed_loop.py on a saved log (blocks until the
    plot window is closed)."""
    plot_script = PROJECT_ROOT / 'scripts' / 'plot_cpp_closed_loop.py'
    subprocess.run([sys.executable, str(plot_script), str(path)])


def sender_loop(conn, qube, period: float, stop_event: threading.Event, log: ClosedLoopLog = None) -> None:
    # Fixed schedule (next_send += period) so the cadence doesn't drift by the
    # time spent sending; if we fall behind, resync instead of bursting.
    next_send = time.monotonic()
    while not stop_event.is_set():
        state = qube.get_state().tolist()
        t = time.time()
        if log is not None:
            log.record(t, state)
        try:
            send_msg(conn, {'t': t, 'x': state})
        except OSError:
            stop_event.set()
            return
        next_send += period
        remaining = next_send - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            next_send = time.monotonic()


def receiver_loop(conn, qube, stop_event: threading.Event, log: ClosedLoopLog = None) -> None:
    while not stop_event.is_set():
        try:
            msg = recv_msg(conn)
        except OSError:
            stop_event.set()
            return
        if msg is None:
            stop_event.set()
            return
        u = msg.get('u')
        if u:
            qube.set_torque_setpoint(float(u[0]))
            if log is not None:
                log.last_u = float(u[0])
        solve_ms = msg.get('solve_ms')
        if solve_ms is not None and log is not None:
            log.record_solve(time.time(), float(solve_ms))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpc-config', default=str(PROJECT_ROOT / 'model/furuta_mpc.json'))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=56123)
    parser.add_argument('--sim-rate', type=float, default=None,
                        help='Rate [Hz] at which the simulator integrates the dynamics '
                             '(default: the simulator\'s own rate, QubeBase.frequency = 4000 Hz).')
    parser.add_argument('--state-rate', type=float, default=200.0,
                        help='Rate [Hz] at which the state is sent to the client.')
    parser.add_argument('--dump', default=None,
                        help='Path prefix (bare filename resolved under model/) to dump the client '
                             'session\'s closed-loop log to, as "<dump>.npz". Defaults to the MPC '
                             'config\'s save_path if its save flag is true, else no dump.')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not open scripts/plot_cpp_closed_loop.py on the saved dump.')
    args = parser.parse_args()

    # furuta_mpc.json is the same source scripts/export_mpc_config.py reads
    # dt/p from when it writes model/mpc_config.yaml for the C++ side, so
    # the two sides can't drift apart on dt.
    furuta_mpc = json.loads(Path(args.mpc_config).read_text())
    dt = furuta_mpc['dt']  # MPC model/prediction step; only recorded in the dump
    state_period = 1.0 / args.state_rate
    qube = create_hardware({'target': 'sim', 'p': furuta_mpc['p'], 'frequency': args.sim_rate})
    sim_rate = qube.frequency  # the rate actually used (class default if --sim-rate not given)
    sim_period = 1.0 / sim_rate

    dump_prefix = args.dump
    if dump_prefix is None and furuta_mpc.get('save', False):
        dump_prefix = furuta_mpc['save_path']

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f'Qube server listening on {args.host}:{args.port} '
          f'(sim {sim_rate:g} Hz, state stream {args.state_rate:g} Hz, MPC dt={dt}s)')

    # One experiment per server run: serve a single client session, save its
    # log, shut down (simulator and socket), then plot the saved log.
    saved_path = None
    try:
        conn, addr = server.accept()
        print(f'Client connected: {addr}')
        stop_event = threading.Event()
        log = ClosedLoopLog() if dump_prefix else None
        t_send = threading.Thread(target=sender_loop, args=(conn, qube, state_period, stop_event, log), daemon=True)
        t_recv = threading.Thread(target=receiver_loop, args=(conn, qube, stop_event, log), daemon=True)
        t_send.start()
        t_recv.start()
        t_recv.join()
        t_send.join()
        conn.close()
        print('Client disconnected; stopping the server.')
        if log is not None:
            dump_path = PROJECT_ROOT / 'model' / f'{dump_prefix}.npz'
            if log.save(dump_path, dt, state_period, sim_period):
                saved_path = dump_path
    except KeyboardInterrupt:
        pass
    finally:
        qube.terminate()
        server.close()

    if saved_path is not None and not args.no_plot:
        plot_log(saved_path)


if __name__ == '__main__':
    main()
