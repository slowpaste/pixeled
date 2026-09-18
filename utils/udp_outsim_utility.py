# File: ./utils/udp_outsim_utility.py

import logging
import math
import socket
import struct
import threading
import time


class OutSimReader:
    """Listens for BeamNG's OutSim packets and keeps the car's yaw rate.

    OutGauge, which the rest of the dashboard runs on, carries no motion beyond
    the speed: no lateral acceleration, no vertical, no heading. OutSim is the
    other half of the same settings page and carries the whole rigid body. Only
    one number of it is wanted here, the yaw rate, which multiplied by the speed
    is the lateral acceleration a corner puts through the car - and so what the
    dashboard should lean under.

    BeamNG does not send LFS's 64-byte OutSimPack, whatever the protocol's name
    suggests. It sends 88 bytes of its own, opening with the four characters
    BNG1, and the layout below was read off a running game (0.39.4) rather than
    out of a document:

        offset  0   char[4]   "BNG1"
                4   float[3]  position x, y, z          m
               16   float[3]  velocity x, y, z          m/s
               28   float[3]  acceleration x, y, z      m/s^2, gravity removed
               40   float[3]  up vector x, y, z         unit
               52   float[3]  roll, pitch, yaw          radians
               64   float[3]  roll rate, pitch rate, yaw rate   rad/s
               76   float[3]  unidentified; near zero at rest

    The yaw rate at offset 72 was checked against the heading at offset 60
    differentiated over the same packets: 0.414 rad/s against 0.413. Taking the
    rate directly is better than differentiating an angle - no wrap past north
    to handle, no dependence on when packets happen to arrive - so that is what
    this reads, and the heading is passed on untouched for anything that wants
    it.

    Nothing here is guesswork about whether a game is sending: the magic says
    the packet is BeamNG's, the length says it is this version of it, and
    anything else on the port is somebody else's and is dropped. OutSim is off
    in BeamNG by default (Options > Other > Protocols); with it off this simply
    reports nothing, and the dashboard leans along the car alone.
    """

    MAGIC = b'BNG1'
    _PACKET_SIZE = 88
    _BODY = '<21f'                  # everything after the magic

    # rad/s. A car turning faster than this is being thrown by something, or has
    # been teleported, and either way it is not a corner to lean into.
    YAW_LIMIT = 12.0
    YAW_TAU = 0.05                  # seconds of smoothing; it arrives at ~200Hz

    def __init__(self, ip="127.0.0.1", port=4444):
        """
        :param ip: address to listen on. Defaults to localhost.
        :param port: UDP port, to match BeamNG's OutSim setting.
        """
        self.ip = ip
        self.port = port
        self._data = {
            "pos": (0.0, 0.0, 0.0),
            "vel": (0.0, 0.0, 0.0),
            "accel": (0.0, 0.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "roll": 0.0,
            "pitch": 0.0,
            "heading": 0.0,
            "yaw_rate": 0.0,        # rad/s, smoothed
            # time.monotonic() when the last packet landed, or None if none ever
            # has. Same reasoning as the OutGauge reader: nothing in the packet
            # tells a car standing still from nothing sending.
            "received_at": None,
        }
        self._yaw = 0.0
        self._last_at = None
        self._warned = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()    # so a reader stopped can be started again
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def listen_on(self, ip, port):
        """Move to another address, for a game set up on a different port.

        Called from whoever read the config, before any packet has been heard,
        so there is nothing to lose by dropping the socket and taking another.
        """
        if (ip, port) == (self.ip, self.port):
            return
        self.stop()
        self.ip, self.port = ip, port
        self._last_at = None
        self._warned = False
        self.start()

    def _listen_loop(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((self.ip, self.port))
                sock.settimeout(1.0)
                while not self._stop_event.is_set():
                    try:
                        data, _ = sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    if len(data) == self._PACKET_SIZE and data[:4] == self.MAGIC:
                        self._parse_packet(data)
                    elif not self._warned:
                        self._warned = True
                        logging.warning(
                            "OutSim: %d bytes beginning %r on %s:%d is not a "
                            "BeamNG OutSim packet, ignoring this sender",
                            len(data), data[:4], self.ip, self.port)
        except OSError as e:
            logging.warning("OutSim: not listening on %s:%d (%s)",
                            self.ip, self.port, e)

    def _parse_packet(self, data):
        (px, py, pz, vx, vy, vz, ax, ay, az, ux, uy, uz,
         roll, pitch, heading, roll_rate, pitch_rate, yaw_rate,
         _a, _b, _c) = struct.unpack(self._BODY, data[4:])

        now = time.monotonic()
        if abs(yaw_rate) > self.YAW_LIMIT:
            yaw_rate = 0.0          # a reset or a teleport, not a corner
        gap = None if self._last_at is None else now - self._last_at
        if gap is None or not 0.0 < gap < 0.5:
            self._yaw = yaw_rate    # first packet, or a break in the stream
        else:
            self._yaw += (yaw_rate - self._yaw) * (1.0 - math.exp(-gap / self.YAW_TAU))
        self._last_at = now

        with self._lock:
            self._data.update(
                pos=(px, py, pz), vel=(vx, vy, vz), accel=(ax, ay, az),
                up=(ux, uy, uz), roll=roll, pitch=pitch, heading=heading,
                yaw_rate=self._yaw, received_at=now)

    def get_data(self):
        with self._lock:
            return dict(self._data)


# Singleton instance, started by whoever imports this - which is the BeamNG
# dashboard alone, so nothing listens on the port unless that layout is up.
outsim_reader = OutSimReader()
outsim_reader.start()


def get_motion():
    """The latest OutSim values, or the resting defaults if none has arrived."""
    return outsim_reader.get_data()
