#!/usr/bin/env bash
# Flash firmware onto the Framework 16 LED matrix, over USB, with no unplugging.
#
#   sudo firmware/flash-ledmatrix.sh                       the pixeled build
#   sudo firmware/flash-ledmatrix.sh ledmatrix-0.2.0-official.uf2   roll back
#
# 1. Stops pixeled.service, which holds the serial port, and starts it again at
#    the end - but only if it was running to begin with, so a service stopped
#    for benchmarking stays stopped.
# 2. Sends the firmware's bootloader command (32 AC 02). The RP2040 reboots
#    into its ROM bootloader and shows up as a USB drive labelled RPI-RP2.
#    If the module is already in bootloader mode (DIP2 flipped on), this step
#    is skipped.
# 3. Copies the .uf2 onto that drive. The bootloader flashes it and reboots
#    into the new firmware by itself.
# 4. Waits for the matrix to come back and asks it for its version.
#
# The keyboard modules are RP2040s too and would present the same RPI-RP2
# drive in bootloader mode, so this refuses to go ahead unless exactly one
# such drive appears, and it appears after the matrix was told to reboot.
#
# If a flash leaves the matrix unresponsive, the bootloader lives in the chip's
# ROM and cannot be overwritten: unplug the module, flip DIP2 on, plug it back
# in, and run this again with the official .uf2. Flip DIP2 off afterwards.
set -euo pipefail

SERVICE="pixeled.service"
MATRIX_ID="32ac:0020"
BOOT_LABEL="RPI-RP2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UF2="${1:-$SCRIPT_DIR/ledmatrix-0.2.0-pixeled.uf2}"

die() { echo "✗ $*" >&2; exit 1; }
say() { echo "• $*"; }

[ "$EUID" -eq 0 ] || die "run with sudo: mounting the bootloader drive needs root"

# ── Checks before touching anything ──────────────────────────────────────────

[ -f "$UF2" ] || die "no such file: $UF2"
size=$(stat -c %s "$UF2")
# Every UF2 block is 512 bytes and opens with the same 8-byte magic.
magic=$(head -c 8 "$UF2" | od -An -tx1 | tr -d ' \n')
[ "$magic" = "5546320a57515d9e" ] || die "$UF2 is not a UF2 file"
[ $((size % 512)) -eq 0 ] || die "$UF2 is truncated ($size bytes)"
# Family ID at offset 28 of the first block: RP2040 is 0xe48bff56.
family=$(od -An -tx4 -j 28 -N 4 --endian=little "$UF2" | tr -d ' ')
[ "$family" = "e48bff56" ] || die "$UF2 is not RP2040 firmware (family 0x$family)"

matrix_tty() {
  local dev props
  for dev in /dev/ttyACM*; do
    [ -e "$dev" ] || continue
    props=$(udevadm info -q property -n "$dev" 2>/dev/null) || continue
    if grep -qx "ID_VENDOR_ID=${MATRIX_ID%:*}" <<<"$props" \
        && grep -qx "ID_MODEL_ID=${MATRIX_ID#*:}" <<<"$props"; then
      echo "$dev"
      return 0
    fi
  done
  return 1
}

boot_drives() {
  lsblk -rno NAME,LABEL | awk -v l="$BOOT_LABEL" '$2 == l { print "/dev/" $1 }'
}

wait_for() {  # seconds, then a command that succeeds once the wait is over
  local deadline=$((SECONDS + $1))
  shift
  until "$@"; do
    [ $SECONDS -lt $deadline ] || return 1
    sleep 0.25
  done
}

one_boot_drive() { [ "$(boot_drives | wc -l)" -eq 1 ]; }

# ── Stop the service, and put it back however this exits ─────────────────────

was_active=false
if systemctl is-active --quiet "$SERVICE"; then
  was_active=true
  say "stopping $SERVICE"
  systemctl stop "$SERVICE"
fi

mnt=""
cleanup() {
  if [ -n "$mnt" ]; then
    umount "$mnt" 2>/dev/null || umount -l "$mnt" 2>/dev/null || true
    rmdir "$mnt" 2>/dev/null || true
  fi
  if $was_active; then
    say "starting $SERVICE again"
    systemctl start "$SERVICE" || echo "✗ could not restart $SERVICE" >&2
  fi
}
trap cleanup EXIT

# ── Into the bootloader ──────────────────────────────────────────────────────

existing=$(boot_drives | wc -l)
if tty=$(matrix_tty); then
  [ "$existing" -eq 0 ] || die "a $BOOT_LABEL drive is already present - another RP2040 module (keyboard?) is in bootloader mode; can't tell which is the matrix"
  say "matrix on $tty, asking it to reboot into the bootloader"
  stty -F "$tty" 115200 raw -echo
  printf '\x32\xac\x02' > "$tty"
elif [ "$existing" -eq 1 ]; then
  say "no matrix firmware running, but one $BOOT_LABEL drive is present - assuming the matrix is in bootloader mode (DIP2)"
else
  die "LED matrix ($MATRIX_ID) not found, and no single $BOOT_LABEL drive to flash instead"
fi

say "waiting for the $BOOT_LABEL drive"
wait_for 20 one_boot_drive || die "no $BOOT_LABEL drive appeared (found $(boot_drives | wc -l))"
drive=$(boot_drives)
say "bootloader drive is $drive"

# ── Flash ────────────────────────────────────────────────────────────────────

# The desktop may auto-mount the drive at the same time; mounting it here as
# well is harmless, and doesn't depend on whether or where that happened.
mnt=$(mktemp -d /tmp/rpi-rp2.XXXXXX)
wait_for 5 mount "$drive" "$mnt" || die "could not mount $drive"
[ -f "$mnt/INFO_UF2.TXT" ] || die "$drive does not look like an RP2040 bootloader drive"
sed 's/^/    /' "$mnt/INFO_UF2.TXT"

say "copying $(basename "$UF2") ($size bytes)"
# The bootloader reboots as soon as it has the last block, which can pull the
# drive out from under the sync - so a failure here is not yet a failed flash.
cp "$UF2" "$mnt/" && sync || true
umount "$mnt" 2>/dev/null || umount -l "$mnt" 2>/dev/null || true
rmdir "$mnt" 2>/dev/null || true
mnt=""

# ── Check what came back ─────────────────────────────────────────────────────

say "waiting for the matrix to come back"
wait_for 30 matrix_tty >/dev/null || die "the matrix did not come back within 30s - see the recovery note at the top of this script"
tty=$(matrix_tty)
sleep 1   # let it finish enumerating before talking to it

stty -F "$tty" 115200 raw -echo
exec 3<>"$tty"
printf '\x32\xac\x20' >&3
reply=$(timeout 3 head -c 32 <&3 | od -An -tu1 -w32) || true
exec 3>&-

read -r major minor_patch pre _ <<<"$reply"
if [ -z "${major:-}" ]; then
  echo "✗ flashed, and the matrix is back on $tty, but it did not answer the version query" >&2
  exit 1
fi
version="$major.$((minor_patch >> 4)).$((minor_patch & 15))"
# Both builds report 0.2.0; the pixeled one also sets the pre-release flag.
if [ "$pre" = "1" ]; then
  say "done: firmware $version with the pre-release flag set - the pixeled build"
else
  say "done: firmware $version without the pre-release flag - the official build"
fi
