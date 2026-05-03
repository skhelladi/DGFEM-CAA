#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_CSV="${ROOT_DIR}/benchmarks/baseline.csv"
BIN="${ROOT_DIR}/build-mpi/bin/dgalerkin"
RANKS_LIST="1,2,4"
QUICK=0
CASES_LIST="tests/square.json,tests/cube.json,tests/cube_unstr.json"

usage() {
    cat <<'EOF'
Usage: scripts/baseline_mpi.sh [options]

Options:
  --bin PATH        Solver binary (default: build-mpi/bin/dgalerkin)
  --out PATH        Output CSV (default: benchmarks/baseline.csv)
  --ranks LIST      Comma-separated ranks list (default: 1,2,4)
    --cases LIST      Comma-separated config list (default: square,cube,cube_unstr)
  --quick           Only run 1 rank on each baseline case
  -h, --help        Show this help

Baseline cases:
  tests/square.json
  tests/cube.json
  tests/cube_unstr.json
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)
            BIN="$2"
            shift 2
            ;;
        --out)
            OUT_CSV="$2"
            shift 2
            ;;
        --ranks)
            RANKS_LIST="$2"
            shift 2
            ;;
        --cases)
            CASES_LIST="$2"
            shift 2
            ;;
        --quick)
            QUICK=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -x "$BIN" ]]; then
    echo "Binary not executable: $BIN" >&2
    echo "Build MPI target first (cmake -DDG_USE_MPI=ON ...)." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"

if [[ ! -f "$OUT_CSV" ]]; then
    echo "timestamp,case_name,ranks,steps,wall_s,startup_s,step_s,max_rss_kb,exit_code,binary,config" > "$OUT_CSV"
fi

IFS=',' read -r -a RANKS <<< "$RANKS_LIST"
if [[ $QUICK -eq 1 ]]; then
    RANKS=(1)
fi

IFS=',' read -r -a CASES <<< "$CASES_LIST"

json_number() {
    local file="$1"
    local key="$2"
    awk -F: -v key="$key" '
        $0 ~ "\"" key "\"" {
            v=$2
            gsub(/[ ,]/, "", v)
            print v
            exit
        }
    ' "$file"
}

extract_prof_value() {
    local prof_file="$1"
    local section="$2"
    awk -F',' -v sec="$section" '
        $1=="PROF_CSV" && $2=="0" && $3==sec { print $4; found=1; exit }
        END { if (!found) print 0 }
    ' "$prof_file"
}

extract_prof_value_any() {
    local prof_file="$1"
    shift
    local key
    for key in "$@"; do
        local v
        v="$(extract_prof_value "$prof_file" "$key")"
        if [[ "$v" != "0" ]]; then
            echo "$v"
            return 0
        fi
    done
    echo 0
}

for cfg_rel in "${CASES[@]}"; do
    cfg_path="${ROOT_DIR}/${cfg_rel}"
    if [[ ! -f "$cfg_path" ]]; then
        echo "Skip missing config: $cfg_path" >&2
        continue
    fi

    start_t="$(json_number "$cfg_path" start)"
    end_t="$(json_number "$cfg_path" end)"
    step_t="$(json_number "$cfg_path" step)"
    steps="$(awk -v a="$start_t" -v b="$end_t" -v h="$step_t" 'BEGIN { if (h<=0) { print 0; exit } printf "%d", ((b-a)/h)+0.5 }')"

    for ranks in "${RANKS[@]}"; do
        ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        case_name="$(basename "$cfg_rel" .json)"

        run_log="${ROOT_DIR}/benchmarks/${case_name}_r${ranks}.log"
        time_log="${ROOT_DIR}/benchmarks/${case_name}_r${ranks}.time"
        prof_log="${ROOT_DIR}/benchmarks/${case_name}_r${ranks}.prof"

        rm -f "$run_log" "$time_log" "$prof_log"

        echo "[baseline] case=${case_name} ranks=${ranks}"
        set +e
        DG_PROFILE_PHASES=1 /usr/bin/time -l mpirun -n "$ranks" "$BIN" "$cfg_path" > "$run_log" 2> "$time_log"
        exit_code=$?
        set -e

        grep '^PROF_CSV,' "$time_log" > "$prof_log" || true

        wall_s="$(awk '$2=="real" {print $1; exit}' "$time_log")"
        if [[ -z "$wall_s" ]]; then
            wall_s="0"
        fi

        max_rss_kb="$(awk '/maximum resident set size/ {print $1; exit}' "$time_log")"
        if [[ -z "$max_rss_kb" ]]; then
            max_rss_kb="0"
        fi

        cfg_total="$(extract_prof_value "$prof_log" "config_json::TOTAL")"
        mesh_total="$(extract_prof_value "$prof_log" "mesh::TOTAL")"
        precompute_mass="$(extract_prof_value_any "$prof_log" "solver_rk4::precomputeMassMatrix" "solver_euler::precomputeMassMatrix")"
        source_locator="$(extract_prof_value_any "$prof_log" "solver_rk4::sourceLocator" "solver_euler::sourceLocator")"
        observer_locator="$(extract_prof_value_any "$prof_log" "solver_rk4::observerLocator" "solver_euler::observerLocator")"
        initial_halo="$(extract_prof_value_any "$prof_log" "solver_rk4::initialHalo" "solver_euler::initialHalo")"
        precompute_flux="$(extract_prof_value_any "$prof_log" "solver_rk4::precomputeFlux" "solver_euler::precomputeFlux")"

        startup_s="$(awk -v a="$cfg_total" -v b="$mesh_total" -v c="$precompute_mass" -v d="$source_locator" -v e="$observer_locator" -v f="$initial_halo" -v g="$precompute_flux" 'BEGIN {printf "%.6f", (a+b+c+d+e+f+g)}')"
        step_s="$(awk -v w="$wall_s" -v n="$steps" 'BEGIN { if (n<=0) {print 0; exit} printf "%.9f", w/n }')"

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$ts" "$case_name" "$ranks" "$steps" "$wall_s" "$startup_s" "$step_s" "$max_rss_kb" "$exit_code" "$BIN" "$cfg_rel" \
            >> "$OUT_CSV"

        if [[ "$exit_code" -ne 0 ]]; then
            echo "[baseline] FAILED case=${case_name} ranks=${ranks} (exit=${exit_code})" >&2
        fi
    done
done

echo "Baseline CSV updated: $OUT_CSV"