import socket
import struct
import sys

def main():
    # Configuration
    UDP_IP = "127.0.0.1"       # Listen on localhost
    UDP_PORT = 8888            # Port to listen on (match BeamNG.drive OutGauge settings)

    # Define the OutGauge UDP packet structure (96 bytes total)
    # '<' indicates little-endian
    # I    -> unsigned int (4 bytes)        : time
    # 4s   -> char[4]    (4 bytes)         : car name
    # H    -> unsigned short (2 bytes)      : flags
    # c    -> char (1 byte)                 : gear
    # c    -> char (1 byte)                 : plid
    # fffffff -> 7 floats (28 bytes total)  : speed, rpm, turbo, engTemp, fuel, oilPressure, oilTemp
    # I    -> unsigned int (4 bytes)        : dashLights
    # I    -> unsigned int (4 bytes)        : showLights
    # fff  -> 3 floats (12 bytes)           : throttle, brake, clutch
    # 16s  -> char[16] (16 bytes)           : display1
    # 16s  -> char[16] (16 bytes)           : display2
    # i    -> int (4 bytes)                 : id (optional)
    packet_format = '<I4sHccfffffffIIfff16s16si'
    packet_size   = struct.calcsize(packet_format)

    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}...")
    print(f"Expected OutGauge packet size: {packet_size} bytes (should be 96).")

    # Create and bind the UDP socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        print("Socket successfully created and bound.")
    except socket.error as e:
        print(f"Socket error: {e}")
        sys.exit(1)

    try:
        while True:
            data, addr = sock.recvfrom(1024)  # Receive up to 1024 bytes
            length = len(data)
            if length < packet_size:
                print(f"Received incomplete packet ({length} bytes) from {addr}.")
                continue

            # Unpack the binary data according to the format
            unpacked_data = struct.unpack(packet_format, data[:packet_size])

            (
                time_ms,            # unsigned int
                car_name_bytes,     # 4-byte string
                flags,              # unsigned short
                gear_byte,          # char
                plid_byte,          # char
                speed,              # float
                rpm,                # float
                turbo,              # float
                engTemp,            # float
                fuel,               # float
                oilPressure,        # float
                oilTemp,            # float
                dashLights,         # unsigned int
                showLights,         # unsigned int
                throttle,           # float
                brake,              # float
                clutch,             # float
                display1_bytes,     # 16-byte string
                display2_bytes,     # 16-byte string
                id_optional         # int
            ) = unpacked_data

            # Decode text fields
            car_name  = car_name_bytes.decode(errors='ignore').strip('\x00')
            display1  = display1_bytes.decode(errors='ignore').strip('\x00')
            display2  = display2_bytes.decode(errors='ignore').strip('\x00')

            # gear_byte and plid_byte are single bytes, convert them to int
            gear = gear_byte[0] if isinstance(gear_byte, bytes) else ord(gear_byte)
            plid = plid_byte[0] if isinstance(plid_byte, bytes) else ord(plid_byte)

            # Print the parsed data
            print("\n--- Received OutGauge Packet ---")
            print(f"Source Address    : {addr}")
            print(f"time_ms           : {time_ms}")
            print(f"car_name          : {car_name}")
            print(f"flags             : {flags}")
            print(f"gear              : {gear}")
            print(f"plid              : {plid}")
            print(f"speed             : {speed:.5f} m/s")
            print(f"rpm               : {rpm:.2f}")
            print(f"turbo             : {turbo:.2f} BAR")
            print(f"engTemp           : {engTemp:.2f} °C")
            print(f"fuel              : {fuel:.3f} (ratio of full)")
            print(f"oilPressure       : {oilPressure:.2f} BAR")
            print(f"oilTemp           : {oilTemp:.2f} °C")
            print(f"dashLights        : {dashLights}")
            print(f"showLights        : {showLights}")
            print(f"throttle          : {throttle:.3f} (0-1)")
            print(f"brake             : {brake:.3f} (0-1)")
            print(f"clutch            : {clutch:.3f} (0-1)")
            print(f"display1          : {display1}")
            print(f"display2          : {display2}")
            print(f"id (optional)     : {id_optional}")
            print("---------------------------------")

    except KeyboardInterrupt:
        print("\nExiting by user request.")

    # Close socket
    sock.close()
    print("Socket closed.")

if __name__ == "__main__":
    main()
