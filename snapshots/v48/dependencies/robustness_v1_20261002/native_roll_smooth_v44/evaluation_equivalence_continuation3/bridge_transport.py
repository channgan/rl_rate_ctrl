import socket,secrets
from pathlib import Path

def choose_bridge_port(candidates=None):
    low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    ports=[p for p in range(20000,30000) if not low<=p<=high]
    if not ports:raise RuntimeError('No configured non-ephemeral bridge port pool')
    attempts=(secrets.choice(ports) for _ in range(64)) if candidates is None else iter(candidates)
    for port in attempts:
        if port not in ports:raise ValueError('Bridge port outside non-ephemeral pool')
        try:
            with socket.socket() as probe:probe.bind(('127.0.0.1',port))
            return port
        except OSError:continue
    raise RuntimeError('No free bridge port; do not start simulator')

def bridge_ready(log):
    text=Path(log).read_text(errors='replace') if Path(log).exists() else ''
    if 'RL bridge bind failed' in text:raise RuntimeError('RL bridge bind failed before INIT; preserve failed attempt')
    return 'RL bridge ready on localhost:' in text
