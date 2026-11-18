#!/usr/bin/env bash
# Install the onboard runner on a Raspberry Pi 5 running 64-bit Raspberry Pi OS (Bookworm or later).
#
# Run from the repo copy on the Pi:   sudo tools/onboard/install-pi.sh [--user NAME] [--repo DIR]
# Safe to run again: it only adds what is missing. Reboot afterwards so the UART setting takes effect.
#
# What it sets up
#   - apt packages, a venv at /opt/uavloc/venv with onnxruntime numpy opencv pillow rasterio pymavlink,
#     and `uavloc` itself from this repo (no torch: the onboard code does not import it)
#   - the UART on GPIO 14/15 as /dev/ttyAMA0 (docs/ardupilot-integration.md section 4), serial console off
#   - /home/<user>/missions (copy mission packages here) and /home/<user>/flight-logs
#   - the systemd service uavloc-onboard, enabled, started once a package is in the missions folder
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_USER="${SUDO_USER:-pi}"
INSTALL_DIR="/opt/uavloc"
VENV_DIR="${INSTALL_DIR}/venv"
UNIT_NAME="uavloc-onboard"
UNIT_TARGET="/etc/systemd/system/${UNIT_NAME}.service"
DEFAULTS_FILE="/etc/default/${UNIT_NAME}"
UART_LINE="dtparam=uart0"
PIP_PACKAGES=(onnxruntime numpy opencv-python-headless rasterio pymavlink pillow)
APT_PACKAGES=(python3-venv python3-pip python3-dev build-essential libxml2-dev libxslt1-dev)

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

while (($# > 0)); do
  case "$1" in
    --user) RUN_USER="${2:?--user needs a name}"; shift 2 ;;
    --repo) REPO_DIR="${2:?--repo needs a directory}"; shift 2 ;;
    -h | --help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo "$0" --user "${RUN_USER}" --repo "${REPO_DIR}"
fi

[[ "$(uname -m)" == "aarch64" ]] || die "expected a 64-bit OS (aarch64), got $(uname -m)"
id "${RUN_USER}" >/dev/null 2>&1 || die "user ${RUN_USER} does not exist (use --user)"
[[ -f "${REPO_DIR}/pyproject.toml" ]] || die "no pyproject.toml in ${REPO_DIR} (use --repo)"
[[ -f "${SCRIPT_DIR}/${UNIT_NAME}.service" ]] || die "${UNIT_NAME}.service is missing next to this script"

RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
MISSIONS_DIR="${RUN_HOME}/missions"
LOGS_DIR="${RUN_HOME}/flight-logs"

log "apt packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"

log "virtualenv ${VENV_DIR}"
mkdir -p "${INSTALL_DIR}"
[[ -x "${VENV_DIR}/bin/python" ]] || python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install "${PIP_PACKAGES[@]}"
# --no-deps: the repo's metadata lists nothing, and torch must stay off the Pi
"${VENV_DIR}/bin/python" -m pip install --no-deps --force-reinstall "${REPO_DIR}"
"${VENV_DIR}/bin/python" -c "import uavloc.onboard.cli" || die "uavloc.onboard does not import"

log "UART on GPIO 14/15"
CONFIG_TXT="/boot/firmware/config.txt"
[[ -f "${CONFIG_TXT}" ]] || CONFIG_TXT="/boot/config.txt"
[[ -f "${CONFIG_TXT}" ]] || die "no config.txt under /boot"
if ! grep -qxF "${UART_LINE}" "${CONFIG_TXT}"; then
  printf '\n# uavloc: flight controller link on GPIO 14/15 (/dev/ttyAMA0)\n%s\n' "${UART_LINE}" >>"${CONFIG_TXT}"
fi
if command -v raspi-config >/dev/null 2>&1; then
  raspi-config nonint do_serial_hw 0    # 0 = enable the serial port hardware
  raspi-config nonint do_serial_cons 1  # 1 = disable the login shell on it
fi
CMDLINE_TXT="$(dirname "${CONFIG_TXT}")/cmdline.txt"
if [[ -f "${CMDLINE_TXT}" ]]; then
  sed -i -E 's/ ?console=(serial0|ttyAMA0),[0-9]+//g' "${CMDLINE_TXT}"
fi
usermod -aG dialout "${RUN_USER}"

log "folders ${MISSIONS_DIR} and ${LOGS_DIR}"
install -d -o "${RUN_USER}" -g "${RUN_USER}" "${MISSIONS_DIR}" "${LOGS_DIR}"

log "systemd service ${UNIT_NAME}"
sed -e "s|@USER@|${RUN_USER}|g" -e "s|@VENV@|${VENV_DIR}|g" -e "s|@MISSIONS@|${MISSIONS_DIR}|g" \
    -e "s|@LOGS@|${LOGS_DIR}|g" -e "s|@DEFAULTS@|${DEFAULTS_FILE}|g" \
    "${SCRIPT_DIR}/${UNIT_NAME}.service" >"${UNIT_TARGET}"
if [[ ! -f "${DEFAULTS_FILE}" ]]; then
  cat >"${DEFAULTS_FILE}" <<'DEFAULTS'
# Extra arguments for `python -m uavloc.onboard`.
# --stub localizes with simulator truth (SITL only). There is no real localizer yet, so the service
# stays stopped until this names one.
UAVLOC_ARGS=
# pymavlink URL of the flight controller (TELEM2 at 921600 baud)
UAVLOC_CONNECTION=/dev/ttyAMA0,921600
DEFAULTS
fi
systemctl daemon-reload
systemctl enable "${UNIT_NAME}.service"
systemctl restart "${UNIT_NAME}.service" || log "service not started yet (needs a package and a localizer); fine"

log "done: reboot for the UART setting, then copy a mission zip into ${MISSIONS_DIR}"
