# File: ./utils/udp_motionsim_utility.py

import logging
import socket
import struct
import threading
import time


class MotionSimReader:
    """Listens for BeamNG's MotionSim packets and keeps the car's motion.

    OutGauge, which the rest of the dashboard runs on, carries no motion beyond
    the speed. MotionSim - "Motion Simulator" on the same settings page, the
    protocol BeamNG sends to motion platforms - carries the whole rigid body:
    how hard it is being accelerated, which way up it is, and how it is
    turning.

    The packet is 88 bytes opening with the four characters BNG1, laid out by
    lua/vehicle/protocols/motionSim.lua in the game's own files:

        offset  0   char[4]   "BNG1"
                4   float[3]  position x, y, z                  m, world
               16   float[3]  velocity x, y, z                  m/s, world
               28   float[3]  acceleration x, y, z              m/s^2, the car's
                                                                own axes, gravity
                                                                not included
               40   float[3]  up vector x, y, z                 unit, world
               52   float[3]  roll, pitch, yaw                  radians
               64   float[3]  roll, pitch, yaw rates            rad/s
               76   float[3]  roll, pitch, yaw accelerations    rad/s^2

    Which way round each of those counts is not in any document, and is read
    off lua/vehicle/protocols.lua, which fills them in. The acceleration is
    the car's own sensor with every axis negated, and the sensor's axes are
    the vehicle's: x to the left, y to the rear, z up (ai.lua logs -gy as the
    acceleration forward and gx against the velocity leftward). Negated, that
    is x to the right, y forward and z down, the motion-platform convention
    the file says it is following - and so what this hands on, renamed so
    nothing downstream has to remember it:

        accel     (right, forward, down)   m/s^2

    The angles are the game's own with roll and yaw negated, to the same
    convention: yaw is positive turning right, clockwise seen from above,
    which is the opposite of the left-turn-positive the heading of a
    right-handed, z-up world would give, and the thing an earlier reading of
    this packet had backwards. Roll is positive rolling right, right side
    down, and pitch positive nose up. The yaw rate agrees in sign with the
    yaw differentiated (0.414 rad/s against 0.413, checked on a running game,
    0.39.4).

    Nothing here is guesswork about whether a game is sending: the magic says
    the packet is BeamNG's, the length says it is this version of it, and
    anything else on the port is somebody else's and is dropped. MotionSim is
    off in BeamNG by default (Options > Other > Protocols); with it off this
    simply reports nothing, and the dashboard moves on OutGauge's speed alone.
    """

    MAGIC = b'BNG1'
    _PACKET_SIZE = 88
    _BODY = '<21f'                  # everything after the magic

    def __init__(self, ip="127.0.0.1", port=4444):
        """
        :param ip: address to listen on. Defaults to localhost.
        :param port: UDP port, to match BeamNG's MotionSim setting.
        """
        self.ip = ip
        self.port = port
        self._data = {
            "pos": (0.0, 0.0, 0.0),
            "vel": (0.0, 0.0, 0.0),
            "accel": (0.0, 0.0, 0.0),   # right, forward, down; m/s^2
            "up": (0.0, 0.0, 1.0),
            "roll": 0.0,                # right side down is positive
            "pitch": 0.0,               # nose up is positive
            "yaw": 0.0,                 # turning right is positive
            "rates": (0.0, 0.0, 0.0),   # roll, pitch, yaw, rad/s
            # time.monotonic() when the last packet landed, or None if none ever
            # has. Same reasoning as the OutGauge reader: nothing in the packet
            # tells a car standing still from nothing sending.
            "received_at": None,
        }
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
                            "MotionSim: %d bytes beginning %r on %s:%d is not a "
                            "BeamNG MotionSim packet, ignoring this sender",
                            len(data), data[:4], self.ip, self.port)
        except OSError as e:
            logging.warning("MotionSim: not listening on %s:%d (%s)",
                            self.ip, self.port, e)

    def _parse_packet(self, data):
        (px, py, pz, vx, vy, vz, ax, ay, az, ux, uy, uz,
         roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate,
         _roll_acc, _pitch_acc, _yaw_acc) = struct.unpack(self._BODY, data[4:])
        with self._lock:
            self._data.update(
                pos=(px, py, pz), vel=(vx, vy, vz), accel=(ax, ay, az),
                up=(ux, uy, uz), roll=roll, pitch=pitch, yaw=yaw,
                rates=(roll_rate, pitch_rate, yaw_rate),
                received_at=time.monotonic())

    def get_data(self):
        with self._lock:
            return dict(self._data)


# Singleton instance, started by whoever imports this - which is the BeamNG
# dashboard alone, so nothing listens on the port unless that layout is up.
motionsim_reader = MotionSimReader()
motionsim_reader.start()


def get_motion():
    """The latest MotionSim values, or the resting defaults if none has arrived."""
    return motionsim_reader.get_data()
