#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-vllm-graph-dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/vllm"

SITE_PACKAGES="$(conda run -n "${ENV_NAME}" python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
DST_DIR="${SITE_PACKAGES}/vllm"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "source tree not found: ${SRC_DIR}" >&2
  exit 1
fi

if [[ ! -d "${DST_DIR}" ]]; then
  echo "destination package not found: ${DST_DIR}" >&2
  exit 1
fi

count=0
while IFS= read -r -d '' file; do
  rel="${file#${SRC_DIR}/}"
  target="${DST_DIR}/${rel}"
  mkdir -p "$(dirname "${target}")"
  rm -f "${target}"
  ln -s "${file}" "${target}"
  count=$((count + 1))
done < <(find "${SRC_DIR}" -type f ! -name '*.so' ! -path '*/__pycache__/*' -print0)

echo "Synced ${count} non-.so files from ${SRC_DIR} to ${DST_DIR}"
