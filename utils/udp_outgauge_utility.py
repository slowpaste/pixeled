# File: ./utils/udp_outgauge_utility.py

import socket
import struct
import threading

class OutGaugeReader:
    """
    A utility class that listens for OutGauge UDP packets and stores the latest
    telemetry data. Other modules can retrieve this data via `get_data()`.
    """

    # 96-byte OutGauge packet format (BeamNG variant)
    _PACKET_FORMAT = '<I4sHccfffffffIIfff16s16si'
    _PACKET_SIZE = struct.calcsize(_PACKET_FORMAT)

    def __init__(self, ip="127.0.0.1", port=8888):
        """
        :param ip: IP address to listen on. Defaults to localhost.
        :param port: UDP port. Must match BeamNG OutGauge settings (Options > Other > Protocols).
        """
        self.ip = ip
        self.port = port

        # Dictionary to store the latest telemetry values
        # e.g. {"speed": float, "rpm": float, "turbo": float, ...}
        self._data = {
            "time_ms": 0,
            "car_name": "beam",
            "flags": 0,
            "gear": 0,
            "plid": 0,
            "speed": 0.0,
            "rpm": 0.0,
            "turbo": 0.0,
            "engTemp": 0.0,
            "fuel": 1.0,
            "oilPressure": 0.0,
            "oilTemp": 0.0,
            "dashLights": 0,
            "showLights": 0,
            "throttle": 0.0,
            "brake": 0.0,
            "clutch": 0.0,
            "id": 0
        }

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """
        Starts the background thread that listens for UDP packets.
        """
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """
        Signals the listening thread to stop and waits for it to exit.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def _listen_loop(self):
        """
        The main loop that binds to the UDP port and continuously reads OutGauge packets.
        """
        print(f"[OutGaugeReader] Starting UDP listener on {self.ip}:{self.port}")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((self.ip, self.port))
                sock.settimeout(1.0)  # 1-second timeout to check stop_event periodically

                while not self._stop_event.is_set():
                    try:
                        data, addr = sock.recvfrom(1024)
                        if len(data) == self._PACKET_SIZE:
                            self._parse_packet(data)
                    except socket.timeout:
                        # Timeout reached; check stop event again
                        pass
        except OSError as e:
            print(f"[OutGaugeReader] Socket error: {e}")

        print("[OutGaugeReader] UDP listener stopped.")

    def _parse_packet(self, data):
        """
        Unpacks the 96-byte OutGauge data and stores it in self._data.
        """
        unpacked_data = struct.unpack(self._PACKET_FORMAT, data)

        (
            time_ms,
            car_name_bytes,
            flags,
            gear_byte,
            plid_byte,
            speed,
            rpm,
            turbo,
            engTemp,
            fuel,
            oilPressure,
            oilTemp,
            dashLights,
            showLights,
            throttle,
            brake,
            clutch,
            display1_bytes,
            display2_bytes,
            id_optional
        ) = unpacked_data

        car_name = car_name_bytes.decode(errors="ignore").strip('\x00')
        gear = gear_byte[0] if isinstance(gear_byte, bytes) else ord(gear_byte)
        plid = plid_byte[0] if isinstance(plid_byte, bytes) else ord(plid_byte)

        with self._lock:
            self._data["time_ms"]     = time_ms
            self._data["car_name"]    = car_name
            self._data["flags"]       = flags
            self._data["gear"]        = gear
            self._data["plid"]        = plid
            self._data["speed"]       = speed
            self._data["rpm"]         = rpm
            self._data["turbo"]       = turbo
            self._data["engTemp"]     = engTemp
            self._data["fuel"]        = fuel
            self._data["oilPressure"] = oilPressure
            self._data["oilTemp"]     = oilTemp
            self._data["dashLights"]  = dashLights
            self._data["showLights"]  = showLights
            self._data["throttle"]    = throttle
            self._data["brake"]       = brake
            self._data["clutch"]      = clutch
            self._data["id"]          = id_optional

    def get_data(self):
        """
        Returns a copy of the current telemetry dictionary (thread-safe).
        """
        with self._lock:
            return dict(self._data)

# Singleton Instance
outgauge_reader = OutGaugeReader()
outgauge_reader.start()

def get_telemetry():
    """
    Convenience function to retrieve the latest telemetry data.
    """
    return outgauge_reader.get_data()
