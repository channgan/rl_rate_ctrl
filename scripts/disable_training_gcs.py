"""Disable the default GCS link only for RL training instances."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--px4', type=Path, required=True)
    args = parser.parse_args()
    path = args.px4 / 'ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink'
    source = path.read_text()
    marker = '# RL training: no default UDP14550 ground-station link'
    if marker in source:
        return
    start = source.index('# GCS link\n')
    end = source.index('# API/Offboard link', start)
    block = source[start:end]
    if 'mavlink start -x -u $udp_gcs_port_local' not in block:
        raise RuntimeError('Unexpected PX4 GCS startup section')
    path.write_text(source[:start] + marker + '\nif [ -z "$PX4_RL_PORT" ]; then\n'
                    + block + 'fi\n\n' + source[end:])


if __name__ == '__main__':
    main()
