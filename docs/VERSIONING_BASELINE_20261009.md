# Current-state baseline, 2026-10-09

This is one honest baseline of accumulated local training work, based on commit
63558d3. It is not a reconstruction of per-iteration commits or evidence of
historical pushes. Existing training-related edits are preserved as found; their
individual authorship is not inferred. Unrelated interface/NMPC notes are omitted.

The public Python package, relevant tests and startup documentation are included.
`snapshots/v48` records the actual backtracking_revision3 training/development
Python code, configuration/templates, preparation/diagnosis/recovery scripts,
historical Python dependencies and the qualified sync2 native source changes.
`source_manifest.json` maps their source and snapshot SHA256 values. Source files
were read twice to check stability. This operation never edits the running tree.

Local machine paths are replaced by named configuration tokens. Use
`scripts/materialize_snapshot.py --paths /private/paths.json --out /new/offline/tree`
to render a separate tree. Supply forward-slash values for DRL_ROOT_WINDOWS,
DRL_ROOT_LINUX, WINDOWS_HOME, WINDOWS_HOME_LINUX, LINUX_HOME,
WORK_SCRIPTS_WINDOWS and WORK_SCRIPTS_LINUX. The paths file is private and must not
be committed. Rendering does not run training or rewrite the source checkout.

This is source/configuration preservation, not a self-contained simulator image.
PX4/Gazebo, the Python environment, historical model A, guard data, captures and
checkpoint inputs remain external. Existing runtime hashes refer to original
executed files. Rendering paths changes bytes: do not treat rendered historical
admission contracts as newly authorized or silently regenerate their hashes.
Any new run requires its own reviewed paths, dependency verification and contract.

No raw telemetry, logs, ULogs, model binaries, checkpoints, canonical accounting
ledgers, credentials or private path mapping are included. Those local artifacts
are NOT backed up by this Git commit. Model IDs/checksums and compact outcome
summaries provide references only.

V48 includes three explicit protocol revisions: adaptive smooth weighting from
the first update; historical memory quantiles diagnostic in every batch; and
fixed finite backtracking through 1/4096 with stricter float64 bias checks.
Do not attribute its outcomes solely to the learning-rate change. Rejection
restores the batch-start actor/optimizer/RNG; the latest accepted checkpoint is
separate. The step17 continuation retains step10 as its batch comparison anchor.
Only a verified final update40 may enter the frozen54 development matrix.

From this baseline onward, each completed iteration should commit reviewed code,
tests, configuration, compact results and recovery notes to a dedicated branch,
then push without force and verify the remote hash. Never claim remote backup
before that verification. Do not capture moving runtime files or secrets.

Validation: 286 offline project tests passed on the isolated rendered LF checkout (62.66s), including the path materializer. No simulator or native training calls were started by these tests. See VALIDATION_SUMMARY.json.
