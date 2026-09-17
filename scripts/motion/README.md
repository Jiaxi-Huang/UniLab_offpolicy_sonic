# Motion Preprocessing

This directory contains scripts for preprocessing motion data for motion tracking tasks.

## SONIC Benchmark Compare

`output_report/export_sonic_compare.py` exports clip-paired side-by-side videos plus the
machine-readable evidence needed for benchmark reports. Release mode compares
the local policy with the SONIC official release; backend mode runs the same
trained policy in MuJoCo and Motrix and renders both physics snapshots through
the MuJoCo visual model, so visible differences are backend behavior rather
than renderer differences.

```bash
uv run output_report/export_sonic_compare.py \
  --mode release \
  --trained-checkpoint logs/flash_sac/G1SonicManager/<run>/model_150000.pt \
  --release-checkpoint data/sonic_release/last.pt \
  --num-clips 10 --seed 20260920 \
  --output benchmarks/sonic_compare/release

uv run output_report/export_sonic_compare.py \
  --mode backend \
  --trained-checkpoint logs/flash_sac/G1SonicManager/<run>/model_150000.pt \
  --release-checkpoint logs/flash_sac/G1SonicManager/<run>/model_150000.pt \
  --num-clips 10 --seed 20260920 \
  --output benchmarks/sonic_compare/backend
```

Each output directory contains `gallery/clip-*/clip.mp4` (1080p, 50 fps),
`preview.jpg`, `meta.json`, per-frame `frame_metrics.json`, `curves.npz`, and
a top-level `manifest.json`, `gallery/index.json`, and `clip_metrics.csv`.
The manifest records the checkpoint paths, runtime, and git commit so gallery
items remain reproducible.

For metric JSON produced by `unilab-run-bench-sonic-metric`, render a
reference-style report with:

```bash
uv run output_report/report_sonic_compare.py \
  benchmarks/sonic_metrics.json --output-dir benchmarks/sonic_report
```

After exporting a video bundle, add its per-frame curve to the report:

```bash
uv run output_report/report_sonic_compare.py \
  benchmarks/sonic_metrics.json --gallery-bundle benchmarks/sonic_compare/release \
  --output-dir benchmarks/sonic_report
```

The report script emits dark-theme PNG/PDF summary charts, clip-level CSV,
a Markdown table, and, when given the export bundle directory alongside the
metric JSON, a per-frame error PDF/PNG.

## Mocap vs iLQR Comparison Replay

The `replay_mocap_ilqr.py` script plays an iLQR-converted SONIC clip as the
solid robot while the source mocap is drawn as a translucent ghost,
frame-locked — the drift between the dynamically feasible plan and the
original retarget is directly visible.  It also prints a numeric deviation
summary (joint/root deltas) at startup.

```bash
# Overlay comparison (default: ghost in place)
uv run scripts/motion/replay_mocap_ilqr.py \
  --original data/lafan1/robot_filtered/dance1_subject1.npz \
  --ilqr data/lafan1_ilqr/robot_filtered/dance1_subject1.npz

# Side-by-side view instead
uv run scripts/motion/replay_mocap_ilqr.py --original ... --ilqr ... --ghost-offset 1.2 0

# Headless check without a viewer (builds both robots, prints deviations)
uv run scripts/motion/replay_mocap_ilqr.py --original ... --ilqr ... --dry-run
```

Controls: `Space` pauses; `--speed 0.5` for slow motion, `--loop` to loop,
`--start-frame N` to jump in.

## LAFAN1 iLQR Dynamics Resolution

The `mocap2ilqr.py` script re-solves the converted LAFAN1 dataset
(`data/lafan1`, produced by `unilab.tools.lafan_data`) with a
receding-horizon iLQR tracker and writes a derived dataset such as
`data/lafan1_ilqr` with `robot_filtered/`, `smpl_filtered/` (copied
unchanged), `packed/`, and an `ilqr_manifest.json` solve report.  The source
dataset is never modified.  The solver runs on the same scene XMLs and
physics rate as the `g1_sonic` training task (`dt=0.005`, `sub=4`, 50 Hz
control); plans that diverge fall back to the source clip and are flagged in
the manifest.

### Usage

```bash
# Full dataset (long: ~15-20 core-hours; prefer --jobs 2-4)
uv run scripts/motion/mocap2ilqr.py --source data/lafan1 --output data/lafan1_ilqr --jobs 4

# High-dynamic families first
uv run scripts/motion/mocap2ilqr.py --source data/lafan1 --output data/lafan1_ilqr \
  --clips run1_subject1,run1_subject2,dance1_subject1

# Re-run only missing clips (existing outputs are skipped), then repack
uv run scripts/motion/mocap2ilqr.py --source data/lafan1 --output data/lafan1_ilqr --jobs 4
```

### Notes

- `--tracked-bodies` defaults to the 11-body dbm set; `sonic14` tracks all
  SONIC bodies including the wrists (better for dance/fight arm motion).
- `--max-joint-deviation` / `--max-root-deviation` bound how far a plan may
  drift from the mocap reference before the clip falls back to the source.
- The A/B training owners are `task=g1_sonic/mujoco_ilqr` (no BC) and
  `task=g1_sonic/mujoco_ilqr_bc` (reference-mode actor BC); point
  `sonic_benchmark_dataset=lafan1_ilqr` at the derived dataset for per-clip
  evaluation.

## BONES-SEED CSV Replay

The `replay_bones_seed_csv.py` script replays local BONES-SEED G1 CSV clips
directly in the MuJoCo viewer.

### Input Format

The replay script expects a fixed 36-column layout:
- `Frame`
- `root_translateX/Y/Z`
- `root_rotateX/Y/Z`
- 29 `*_joint_dof` columns that map directly to G1 MuJoCo joint names

The script assumes:
- `root_translate*` is in centimeters and converts it to meters
- `root_rotate*` is in degrees
- `*_joint_dof` is in degrees

### Usage

```bash
# Replay the whole flip dataset
uv run scripts/motion/replay_bones_seed_csv.py

# Replay one clip
uv run scripts/motion/replay_bones_seed_csv.py \
  --input src/unilab/assets/motions/g1/flip/flip_090_001__A304.csv

# Validate parsing without opening the viewer
uv run scripts/motion/replay_bones_seed_csv.py --dry-run
```

### Controls

- `Space`: pause / resume
- `[`: previous CSV in playlist
- `]`: next CSV in playlist

## BONES-SEED CSV to NPZ

The `bones_seed_csv_to_npz.py` script converts local G1 flip CSV clips into NPZ
files with precomputed forward kinematics.

### Output Format

Generated NPZ files contain:
- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

### Usage

```bash
# Convert the whole flip dataset into src/unilab/assets/motions/g1/flip_npz
uv run scripts/motion/bones_seed_csv_to_npz.py

# Convert one clip next to a chosen output file
uv run scripts/motion/bones_seed_csv_to_npz.py \
  --input src/unilab/assets/motions/g1/flip/flip_090_001__A304.csv \
  --output temp/flip_090_001__A304.npz

# Validate inputs without exporting
uv run scripts/motion/bones_seed_csv_to_npz.py --dry-run
```

## CSV to NPZ Conversion

The `csv_to_npz.py` script converts motion data from CSV format to NPZ format with precomputed forward kinematics.

### Input Format

CSV files should contain motion data in Unitree's generalized coordinate convention:
- Columns 0-2: Base position (x, y, z)
- Columns 3-6: Base quaternion (x, y, z, w) - will be converted to wxyz internally
- Columns 7+: Joint angles (29 joints for G1)

### Output Format

NPZ files contain:
- `fps`: Frame rate (integer)
- `joint_pos`: Joint positions (N_frames × N_joints)
- `joint_vel`: Joint velocities (N_frames × N_joints)
- `body_pos_w`: Body positions in world frame (N_frames × N_bodies × 3)
- `body_quat_w`: Body quaternions in world frame (N_frames × N_bodies × 4, wxyz)
- `body_lin_vel_w`: Body linear velocities (N_frames × N_bodies × 3)
- `body_ang_vel_w`: Body angular velocities (N_frames × N_bodies × 3)

### Usage

```bash
# Basic usage
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50

# With custom model file
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50 \
  --model_file path/to/model.xml

# Process specific line range
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50 \
  --line_range 100 500
```

### Parameters

- `--input_file`: Path to input CSV file (required)
- `--output_file`: Path to output NPZ file (required)
- `--input_fps`: Frame rate of input CSV (default: 30)
- `--output_fps`: Desired output frame rate (default: 50)
- `--model_file`: MuJoCo model file (default: G1 flat scene)
- `--line_range`: Line range to process [start, end] (optional)

### Notes

- The script uses LERP for position interpolation and SLERP for quaternion interpolation
- Velocities are computed using numerical differentiation
- Forward kinematics is computed using MuJoCo for all bodies
- The output FPS should match the control frequency of your training environment (typically 50 Hz)
