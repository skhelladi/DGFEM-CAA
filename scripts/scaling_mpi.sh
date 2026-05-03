#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_PATH="${ROOT_DIR}/build-mpi/bin/dgalerkin"
OUT_CSV="${ROOT_DIR}/benchmarks/scaling_mpi.csv"
TMP_DIR="${TMPDIR:-/tmp}/dg_scaling_mpi"
mkdir -p "${TMP_DIR}" "$(dirname "${OUT_CSV}")"

CASES=("square:tests/square.json" "cube:tests/cube.json")
RANKS=(1 2 4)
MODES=("on" "off")

usage() {
  echo "Usage: $0 [--cases square,cube] [--ranks 1,2,4] [--modes on,off] [--out path.csv]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases)
      shift
      IFS=',' read -r -a case_names <<< "${1:-}"
      CASES=()
      for name in "${case_names[@]}"; do
        case "${name}" in
          square) CASES+=("square:tests/square.json") ;;
          cube) CASES+=("cube:tests/cube.json") ;;
          cube_unstr) CASES+=("cube_unstr:tests/cube_unstr.json") ;;
          *) echo "Unknown case: ${name}" >&2; exit 2 ;;
        esac
      done
      ;;
    --ranks)
      shift
      IFS=',' read -r -a RANKS <<< "${1:-}"
      ;;
    --modes)
      shift
      IFS=',' read -r -a MODES <<< "${1:-}"
      ;;
    --out)
      shift
      OUT_CSV="$1"
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

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Missing binary: ${BIN_PATH}" >&2
  echo "Build first: cd ${ROOT_DIR}/build-mpi && cmake .. -DDG_USE_MPI=ON && make -j4" >&2
  exit 2
fi

if [[ ! -f "${OUT_CSV}" ]]; then
  echo "timestamp,case,ranks,mode,total_s,halo_s,reductions_s,samples,exit_code,profile_file" > "${OUT_CSV}"
fi

parse_profile() {
  local prof="$1"
  python3 - <<'PY' "$prof"
import re, statistics, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("NA,NA,NA,0")
    raise SystemExit(0)

txt = p.read_text(errors='ignore')
def vals(section):
  return [float(m.group(1)) for m in re.finditer(rf'PROF_CSV,\d+,{re.escape(section)},([0-9eE+.-]+)', txt)]

def vals_any(sections):
  for section in sections:
    v = vals(section)
    if v:
      return v
  return []

def med(v):
    return "NA" if not v else f"{statistics.median(v):.6g}"

tot = vals_any(['solver_rk4::TOTAL', 'solver_euler::TOTAL'])
halo = vals_any(['solver_rk4::haloExchange', 'solver_euler::haloExchange'])
red = vals_any(['solver_rk4::reductions', 'solver_euler::reductions'])
print(f"{med(tot)},{med(halo)},{med(red)},{len(tot)}")
PY
}

echo "[scaling] writing ${OUT_CSV}"

for case_entry in "${CASES[@]}"; do
  case_name="${case_entry%%:*}"
  case_cfg="${case_entry##*:}"

  for ranks in "${RANKS[@]}"; do
    for mode in "${MODES[@]}"; do
      stamp="$(date +%Y-%m-%dT%H:%M:%S)"
      run_id="${case_name}_r${ranks}_${mode}_$(date +%s)"
      prof_file="${TMP_DIR}/${run_id}.prof"
      log_file="${TMP_DIR}/${run_id}.log"

      echo "[scaling] case=${case_name} ranks=${ranks} mode=${mode}"

      set +e
      if [[ "${mode}" == "off" ]]; then
        DG_PROFILE_PHASES=1 DG_PROFILE_SOLVER=1 DG_DISABLE_HALO_OVERLAP=1 mpirun -n "${ranks}" "${BIN_PATH}" "${ROOT_DIR}/${case_cfg}" \
          2>"${prof_file}" 1>"${log_file}"
      else
        DG_PROFILE_PHASES=1 DG_PROFILE_SOLVER=1 mpirun -n "${ranks}" "${BIN_PATH}" "${ROOT_DIR}/${case_cfg}" \
          2>"${prof_file}" 1>"${log_file}"
      fi
      rc=$?
      set -e

      metrics="$(parse_profile "${prof_file}")"
      total_s="${metrics%%,*}"
      rest="${metrics#*,}"
      halo_s="${rest%%,*}"
      rest="${rest#*,}"
      red_s="${rest%%,*}"
      samples="${rest##*,}"

      echo "${stamp},${case_name},${ranks},${mode},${total_s},${halo_s},${red_s},${samples},${rc},${prof_file}" >> "${OUT_CSV}"

      if [[ ${rc} -ne 0 ]]; then
        echo "[scaling] WARN failed run (rc=${rc}) case=${case_name} ranks=${ranks} mode=${mode}" >&2
      fi
    done
  done
done

echo "[scaling] done"