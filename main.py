#!/home/ecca/pixeled/venv/bin/python

import serial
import time
import json
import importlib
from compositor import Compositor
import re
import inspect
import logging

#from utils.udp_outgauge_utility import outgauge_reader  # Import the singleton

logging.basicConfig(filename='/home/ecca/pixeled/service.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def send_command(command_id, parameters, with_response=False):
    try:
        with serial.Serial('/dev/ttyACM0', 115200, timeout=1) as ser:
            command = [0x32, 0xAC, command_id] + parameters
            ser.write(bytearray(command))
            if with_response:
                res = ser.read(32)
                return res
    except serial.SerialException as e:
        logging.error(f"Failed to send command: {e}")

def draw_arbitrary_content(content):
    for col, col_data in enumerate(content):
        if len(col_data) != 34:
            raise ValueError("Each column must contain 34 grayscale values.")
        send_command(0x07, [col] + col_data)
    send_command(0x08, [])

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    width, height = 9, 34

    config = load_config('config.json')
    compositor = Compositor(width, height, config)

    # Add modules based on the configuration
    for mod in config:
        module_name = mod["module"]
        module_height = mod.get("height")
        position = mod.get("position")

        # Convert module name from CamelCase to snake_case
        module_filename = camel_to_snake(module_name)

        # Dynamically import the module class
        module_class = getattr(importlib.import_module(f"modules.{module_filename}"), module_name)

        # Collect parameters for the module constructor
        constructor_args = {}
        if "height" in inspect.signature(module_class).parameters:
            constructor_args["height"] = module_height
        for param in inspect.signature(module_class).parameters:
            if param in mod:
                constructor_args[param] = mod[param]

        # Create an instance of the module with the collected parameters
        module_instance = module_class(**constructor_args)

        # Ensure the module has a height attribute
        if not hasattr(module_instance, 'height'):
            raise ValueError(f"Module {module_name} does not have a height attribute")

        # Add the module to the compositor
        compositor.add_module(module_instance, position)

    # Initialize layout once
    compositor.initialize_layout()

    while True:
        final_image = compositor.render()
        if final_image is not None:
            pixels = list(final_image.getdata())
            content = [[pixels[row * width + col] for row in range(height)] for col in range(width)]
            draw_arbitrary_content(content)
        time.sleep(0.02)

if __name__ == "__main__":
    main()
