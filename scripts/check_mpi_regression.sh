#!/usr/bin/env bash
set -euo pipefail

# Compare residual histories produced by the MPI binary at 1 rank and N ranks.
# The runs are isolated in temporary directories that expose the repository tests/
# and data/ trees through symlinks so existing relative paths remain valid.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_PATH="${ROOT_DIR}/build-mpi/bin/dgalerkin"
CASE_NAME="square"
BASELINE_RANKS=1
MPI_RANKS=2
ATOL=1e-10
RTOL=1e-8
KEEP_WORKDIR=0

usage() {
  echo "Usage: $0 [--case square|cube|cube_unstr] [--baseline-ranks 1] [--mpi-ranks 2] [--atol 1e-10] [--rtol 1e-8] [--keep]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      shift
      CASE_NAME="${1:-}"
      ;;
    --baseline-ranks)
      shift
      BASELINE_RANKS="${1:-}"
      ;;
    --mpi-ranks)
      shift
      MPI_RANKS="${1:-}"
      ;;
    --atol)
      shift
      ATOL="${1:-}"
      ;;
    --rtol)
      shift
      RTOL="${1:-}"
      ;;
    --keep)
      KEEP_WORKDIR=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
 done

case "${CASE_NAME}" in
  square) CASE_CFG="tests/square.json" ;;
  cube) CASE_CFG="tests/cube.json" ;;
  cube_unstr) CASE_CFG="tests/cube_unstr.json" ;;
  *) echo "Unknown case: ${CASE_NAME}" >&2; exit 2 ;;
esac

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Missing binary: ${BIN_PATH}" >&2
  echo "Build first: cd ${ROOT_DIR}/build-mpi && cmake .. -DDG_USE_MPI=ON && make -j4" >&2
  exit 2
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/dg_mpi_regression.XXXXXX")"
trap 'if [[ ${KEEP_WORKDIR} -eq 0 ]]; then rm -rf "${WORKDIR}"; else echo "[mpi-regression] kept workdir: ${WORKDIR}"; fi' EXIT

prepare_run_dir() {
  local run_dir="$1"
  mkdir -p "${run_dir}" "${run_dir}/results"
  ln -sfn "${ROOT_DIR}/tests" "${run_dir}/tests"
  ln -sfn "${ROOT_DIR}/data" "${run_dir}/data"
}

run_case() {
  local run_dir="$1"
  local ranks="$2"
  prepare_run_dir "${run_dir}"
  (
    cd "${run_dir}"
    DG_PROFILE_PHASES=1 DG_PROFILE_SOLVER=1 mpirun -n "${ranks}" "${BIN_PATH}" "${CASE_CFG}" > run.log 2> run.prof
  )
}

echo "[mpi-regression] case=${CASE_NAME} baseline=${BASELINE_RANKS} mpi=${MPI_RANKS}"
run_case "${WORKDIR}/baseline" "${BASELINE_RANKS}"
run_case "${WORKDIR}/mpi" "${MPI_RANKS}"

python3 - <<'PY' "${WORKDIR}/baseline/residuals.csv" "${WORKDIR}/mpi/residuals.csv" "${ATOL}" "${RTOL}" "${CASE_NAME}" "${BASELINE_RANKS}" "${MPI_RANKS}"
import csv
import math
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
mpi_path = Path(sys.argv[2])
atol = float(sys.argv[3])
rtol = float(sys.argv[4])
case_name = sys.argv[5]
baseline_ranks = sys.argv[6]
mpi_ranks = sys.argv[7]

if not baseline_path.exists() or not mpi_path.exists():
    missing = [str(p) for p in (baseline_path, mpi_path) if not p.exists()]
    print(f"[mpi-regression] missing residual file(s): {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(1)

with baseline_path.open() as fh:
    baseline_rows = list(csv.reader(fh, delimiter=';'))
with mpi_path.open() as fh:
    mpi_rows = list(csv.reader(fh, delimiter=';'))

if len(baseline_rows) != len(mpi_rows):
    print(f"[mpi-regression] row-count mismatch: baseline={len(baseline_rows)} mpi={len(mpi_rows)}", file=sys.stderr)
    raise SystemExit(1)

header = baseline_rows[0]
if header != mpi_rows[0]:
    print("[mpi-regression] CSV header mismatch", file=sys.stderr)
    raise SystemExit(1)

max_abs = [0.0] * len(header)
max_rel = [0.0] * len(header)
worst = None

for idx, (b_row, m_row) in enumerate(zip(baseline_rows[1:], mpi_rows[1:]), start=2):
    if len(b_row) != len(m_row):
        print(f"[mpi-regression] row width mismatch at line {idx}", file=sys.stderr)
        raise SystemExit(1)
    for col, (b_val, m_val) in enumerate(zip(b_row, m_row)):
        b = float(b_val)
        m = float(m_val)
        abs_err = abs(b - m)
        scale = max(abs(b), abs(m), 1.0)
        rel_err = abs_err / scale
        max_abs[col] = max(max_abs[col], abs_err)
        max_rel[col] = max(max_rel[col], rel_err)
        if abs_err > atol and rel_err > rtol:
            worst = (idx, header[col], b, m, abs_err, rel_err)
            break
    if worst is not None:
        break

if worst is not None:
    line, col_name, b, m, abs_err, rel_err = worst
    print(
        f"[mpi-regression] FAIL case={case_name} baseline={baseline_ranks} mpi={mpi_ranks} "
        f"line={line} col={col_name} baseline={b:.12e} mpi={m:.12e} abs={abs_err:.3e} rel={rel_err:.3e}",
        file=sys.stderr,
    )
    raise SystemExit(1)

summary = ", ".join(
    f"{name}:abs={abs_v:.3e}:rel={rel_v:.3e}"
    for name, abs_v, rel_v in zip(header, max_abs, max_rel)
)
print(f"[mpi-regression] PASS case={case_name} baseline={baseline_ranks} mpi={mpi_ranks} | {summary}")
PY
