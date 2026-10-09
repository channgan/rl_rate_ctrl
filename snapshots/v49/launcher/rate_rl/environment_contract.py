"""Read-only promotion identity for task settings and explicit SDF assets.

This binds the two stage configurations, world/model SDFs and model.config
files, including transitive model:// includes. It does not hash simulator
binaries, plugins, meshes, logs or the entire PX4 checkout.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import unquote
import xml.etree.ElementTree as ET


ENVIRONMENT_CONTRACT = "promotion_environment_task_sdf_v1"
STAGES = ("ball_rig", "free_flight")
PATH_FIELDS = frozenset({"runtime", "px4", "world", "models", "plugin_dir"})
REQUIRED_PATH_FIELDS = frozenset({"runtime", "px4", "world", "models"})


def _json_value(value):
    """Normalize tuples to JSON lists and reject non-finite configuration."""
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("Environment contract requires finite JSON configuration") from exc


def _physical_config(config):
    return _json_value({key: value for key, value in config.items()
                        if key not in PATH_FIELDS and key != "mode"})


def _path(config, field):
    value = config.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing environment asset path: {field}")
    return Path(value).resolve()


def _read(path):
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Missing or unreadable required environment asset: {path}") from exc


def _read_config(path):
    try:
        value = json.loads(_read(path))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid environment configuration: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Environment configuration must be an object: {path}")
    return value


def _xml(data, path):
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid required SDF/model configuration: {path}") from exc


def _relative_path(value, source):
    path = PurePosixPath(unquote(value))
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsupported asset reference {value!r} in {source}")
    return Path(*path.parts)


class _SdfAssets:
    def __init__(self, config):
        self.models = _path(config, "models")
        self.px4_models = _path(config, "px4") / "Tools/simulation/gz/models"
        for directory in (self.models, self.px4_models):
            if not directory.is_dir():
                raise ValueError(f"Missing required model directory: {directory}")
        self.hashes = {}
        self.visited = set()

    def _key(self, path):
        for prefix, root in (("models", self.models), ("px4_models", self.px4_models)):
            try:
                return f"{prefix}/{path.relative_to(root).as_posix()}"
            except ValueError:
                continue
        raise ValueError(f"Included SDF must belong to a configured model directory: {path}")

    def add(self, path, key=None):
        path = Path(path)
        key = key or self._key(path)
        data = _read(path)
        self.hashes[key] = hashlib.sha256(data).hexdigest()
        resolved = path.resolve()
        if resolved in self.visited:
            return
        self.visited.add(resolved)
        root = _xml(data, path)
        if path.name == "model.config":
            sdf_files = [(node.text or "").strip() for node in root.findall("sdf")]
            if not sdf_files:
                raise ValueError(f"model.config does not declare an SDF: {path}")
            # Bind all declared versions rather than guess Gazebo's selection.
            for name in sdf_files:
                self.add(path.parent / _relative_path(name, path))
        for include in root.findall(".//include"):
            uri = (include.findtext("uri") or "").strip()
            if not uri.startswith("model://"):
                raise ValueError(f"Unsupported or missing SDF include URI {uri!r} in {path}")
            relative = _relative_path(uri[len("model://"):], path)
            # Match GZ_SIM_RESOURCE_PATH: generated training models first.
            for directory in (self.models, self.px4_models):
                model_dir = directory / relative.parts[0]
                if model_dir.is_dir():
                    self.add(model_dir / "model.config")
                    if len(relative.parts) > 1:
                        self.add(directory / relative)
                    break
            else:
                raise ValueError(f"Missing referenced model {uri!r} in {path}")

    def collect(self, config):
        self.add(_path(config, "world"), "world")
        # Include all generated stage model descriptions, not logs/caches.
        paths = sorted(path for path in self.models.rglob("*")
                       if path.is_file() and (path.suffix == ".sdf" or path.name == "model.config"))
        if not paths:
            raise ValueError(f"No required SDF/model assets found in {self.models}")
        for path in paths:
            self.add(path)
        return dict(sorted(self.hashes.items()))


def environment_contract(env):
    """Return a location-independent JSON contract without resetting the env.

    A pure test backend has no runtime/PX4/world/models paths. A real backend
    must provide all four, agree with its on-disk stage configuration, and
    supply both stage configurations and every required transitive SDF asset.
    """
    base = env.unwrapped
    if not is_dataclass(base.config) or isinstance(base.config, type):
        raise ValueError("Environment task configuration must be a dataclass instance")
    config = base.backend.config
    if not isinstance(config, dict):
        raise ValueError("Backend configuration must be a dictionary")
    result = dict(version=ENVIRONMENT_CONTRACT, task=_json_value(asdict(base.config)))
    if not REQUIRED_PATH_FIELDS.intersection(config):
        return dict(result, kind="in_memory", physical=_physical_config(config))
    if not REQUIRED_PATH_FIELDS.issubset(config):
        raise ValueError("Incomplete real environment asset paths")
    stage = config.get("mode")
    if stage not in STAGES:
        raise ValueError("Unknown current environment stage")
    runtime = _path(config, "runtime")
    stage_configs = {name: _read_config(runtime / f"runtime.{name}.json") for name in STAGES}
    for name, candidate in stage_configs.items():
        if candidate.get("mode") != name or _path(candidate, "runtime") != runtime:
            raise ValueError(f"Stage configuration does not identify its stage/runtime: {name}")
        for field in REQUIRED_PATH_FIELDS:
            _path(candidate, field)
    current = stage_configs[stage]
    if _physical_config(config) != _physical_config(current):
        raise ValueError("Live backend physical configuration differs from its on-disk stage configuration")
    for field in PATH_FIELDS:
        if field in config or field in current:
            if _path(config, field) != _path(current, field):
                raise ValueError(f"Live backend asset path differs from its on-disk stage configuration: {field}")
    return dict(result, kind="gazebo_sdf",
                stages={name: dict(physical=_physical_config(candidate),
                                   assets_sha256=_SdfAssets(candidate).collect(candidate))
                        for name, candidate in stage_configs.items()},
                scope="TaskConfig, stage JSON settings, world/model SDF and model.config including model:// includes; excludes binaries, plugins and meshes")
