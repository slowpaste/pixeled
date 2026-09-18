"""Where does the LED matrix's ~17ms per command go?

Each test sends N commands as separate writes (the firmware handles one USB
packet per loop pass), then a Version query, and times until its reply
arrives - so the time covers the device actually consuming every command, not
just the kernel accepting the writes.

  invalid   magic + unknown id: parsed as None, no I2C redraw
  draw_bw   DrawBW frame: full redraw
  stage     StageGreyCol: writes a scratch buffer, still triggers a redraw
  grey      whole greyscale frames: 9 stage + 1 flush
"""
import sys
import time

import serial
from serial.tools import list_ports

MAGIC = bytes((0x32, 0xAC))
N = 300


def find():
    for p in list_ports.comports():
        if p.vid == 0x32AC and p.pid == 0x0020:
            return p.device
    sys.exit('LED matrix not found')


def version(ser):
    ser.reset_input_buffer()
    ser.write(MAGIC + bytes((0x20,)))
    reply = ser.read(32)
    if len(reply) < 32:
        sys.exit(f'no version reply ({len(reply)} bytes)')
    return reply


def timed(ser, name, commands):
    ser.reset_input_buffer()
    start = time.perf_counter()
    for c in commands:
        ser.write(c)
    version(ser)
    elapsed = time.perf_counter() - start
    n = len(commands)
    print(f'{name:8} {n:4} cmds  {elapsed:6.2f}s  {n / elapsed:7.1f} cmd/s  '
          f'{elapsed / n * 1000:6.2f} ms/cmd', flush=True)
    return elapsed / n


def main():
    ser = serial.Serial(find(), 115200, timeout=10)
    # Any command wakes it; a sleeping module fades back in over ~1s first.
    ser.write(MAGIC + bytes((0x06,)) + bytes(39))
    time.sleep(2.0)
    v = version(ser)
    print(f'firmware {v[0]}.{v[1] >> 4}.{v[1] & 0xF} pre={v[2]}')

    blank = MAGIC + bytes((0x06,)) + bytes(39)
    checker = [MAGIC + bytes((0x06,)) + bytes((0x55 if i % 2 else 0xAA,) * 39)
               for i in range(2)]
    col = [MAGIC + bytes((0x07, x)) + bytes((x * 28,) * 34) for x in range(9)]
    flush = MAGIC + bytes((0x08, 0x00))

    results = {}
    for rep in range(2):
        results.setdefault('invalid', []).append(
            timed(ser, 'invalid', [MAGIC + bytes((0x7F,))] * N))
        results.setdefault('draw_bw', []).append(
            timed(ser, 'draw_bw', [checker[i % 2] for i in range(N)]))
        results.setdefault('stage', []).append(
            timed(ser, 'stage', [col[i % 9] for i in range(N)]))
        results.setdefault('grey', []).append(
            timed(ser, 'grey', (col + [flush]) * (N // 10)))
    ser.write(blank)
    print()
    for k, v in results.items():
        ms = min(v) * 1000
        print(f'{k:8} best {ms:6.2f} ms/cmd  -> {1000 / ms:6.1f} cmd/s')


if __name__ == '__main__':
    main()
