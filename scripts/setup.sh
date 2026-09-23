#!/usr/bin/env bash
# Ubuntu 22.04, Gazebo Harmonic and PX4 build prerequisites must be installed.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${RL_RUNTIME_DIR:-$HOME/rl_rate}"
px4="$runtime_dir/PX4-Autopilot"
revision=d6f12ad1c4f70ad3230afd7d86e971421e02fef4
mkdir -p "$runtime_dir"
if [ ! -d "$px4/.git" ]; then
    git clone --branch v1.17.0 --recursive https://github.com/PX4/PX4-Autopilot.git "$px4"
else
    test "$(git -C "$px4" rev-parse HEAD)" = "$revision"
    test -z "$(git -C "$px4" status --porcelain)" || { echo 'Use a new runtime directory; existing PX4 has local changes.' >&2; exit 1; }
fi
test "$(git -C "$px4" rev-parse HEAD)" = "$revision"
python3 -m venv "$runtime_dir/.venv"
python="$runtime_dir/.venv/bin/python"
"$python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$python" -m pip install -e "$project_dir[test]"
"$python" -m pip install -r "$px4/Tools/setup/requirements.txt"
"$python" "$project_dir/scripts/prepare.py" --px4 "$px4" --runtime "$runtime_dir" --mode free_flight
"$python" "$project_dir/scripts/disable_training_gcs.py" --px4 "$px4"
cmake -S "$project_dir/gazebo" -B "$runtime_dir/gazebo-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$runtime_dir/gazebo-build" -j4
(cd "$px4" && PATH="$runtime_dir/.venv/bin:$PATH" make px4_sitl_default -j8)
echo "Ready. Training has not been started. Runtime: $runtime_dir/runtime.free_flight.json"
