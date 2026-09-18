import glob
import logging
import os
import socket
import threading
from collections import deque

NETLINK_KOBJECT_UEVENT = 15
KERNEL_GROUP = 1        # uevents as the kernel sends them, before udev

# The Framework Audio Expansion Card, whose jack the headphones go in.
AUDIO_CARD_VID, AUDIO_CARD_PID = 0x32AC, 0x0010
POWER_SUPPLIES = '/sys/class/power_supply'


class PlugWatch:
    """Something being plugged into the laptop and pulled out, as 'in' and
    'out' events.

    Both of these listen for the kernel's own announcements - a netlink socket
    anyone may read, which wakes the moment the kernel sees the change,
    without polling and without udev, PipeWire or a desktop session in the way.
    A subclass says which announcements are its own, and what they mean about
    whether the thing is plugged in; this only passes on the changes, so
    whatever else the kernel is saying about a device costs nothing.
    """

    def __init__(self):
        self._events = deque()
        self._present = self.plugged_in()
        threading.Thread(target=self._listen, daemon=True).start()

    def plugged_in(self):
        """Whether it is plugged in, as things stand."""
        raise NotImplementedError

    def _state_from(self, fields):
        """What one announcement says: plugged in, not, or nothing to say."""
        raise NotImplementedError

    def _listen(self):
        try:
            sock = socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM,
                                 NETLINK_KOBJECT_UEVENT)
            sock.bind((0, KERNEL_GROUP))
        except OSError as e:
            logging.error(f"Can't watch for {type(self).__name__}: {e}")
            return
        while True:
            try:
                message = sock.recv(16384)
            except OSError:
                continue
            fields = [f.decode('utf-8', 'replace') for f in message.split(b'\0') if f]
            state = self._state_from(fields)
            if state is None or state == self._present:
                continue
            self._present = state
            self._events.append('in' if state else 'out')

    def events(self):
        """What has happened since last asked, oldest first."""
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    def pretend(self, event):
        """Act as if it had been plugged 'in' or pulled 'out', to see what
        that looks like without reaching behind the laptop."""
        self._events.append(event)


class JackWatch(PlugWatch):
    """Headphones going into the jack and coming out.

    The audio expansion card has no jack switch that ALSA or PipeWire can see:
    its one output port reports its availability as unknown. It does something
    better for this: it only shows up on USB while a plug is in it. Pulling the
    headphones is the card disconnecting, and plugging them back is it
    enumerating again, a second or two later, once USB has got round to it. So
    the card's own comings and goings are the jack, picked out by its IDs.
    """

    # How the kernel names a USB device's product: vendor/product/bcdDevice,
    # in lower-case hex without leading zeros.
    PRODUCT = f"PRODUCT={AUDIO_CARD_VID:x}/{AUDIO_CARD_PID:x}/"

    def plugged_in(self):
        for vendor in glob.glob('/sys/bus/usb/devices/*/idVendor'):
            folder = os.path.dirname(vendor)
            try:
                with open(vendor) as v, open(os.path.join(folder, 'idProduct')) as p:
                    if (int(v.read(), 16), int(p.read(), 16)) == (AUDIO_CARD_VID, AUDIO_CARD_PID):
                        return True
            except (OSError, ValueError):
                pass
        return False

    def _state_from(self, fields):
        # Announced once for the device and again for each of its interfaces;
        # only the device itself counts. What it says is taken from the
        # announcement rather than looked up: on the way out the card's sysfs
        # entry can still be there when the kernel says it is going.
        if 'DEVTYPE=usb_device' not in fields:
            return None
        if not any(f.startswith(self.PRODUCT) for f in fields):
            return None
        action = next((f[7:] for f in fields if f.startswith('ACTION=')), None)
        return {'add': True, 'remove': False}.get(action)


class PowerWatch(PlugWatch):
    """The charger going into the USB-C port and coming out.

    Whichever port it is in: what matters is that the laptop is on mains, which
    the kernel says plainly. The announcement itself usually carries the new
    state, and anything else the batteries and ports have to say sends us to
    read it.
    """

    def plugged_in(self):
        for supply in glob.glob(f'{POWER_SUPPLIES}/*'):
            try:
                with open(os.path.join(supply, 'type')) as f:
                    if f.read().strip() != 'Mains':
                        continue
                with open(os.path.join(supply, 'online')) as f:
                    if f.read().strip() == '1':
                        return True
            except OSError:
                pass
        return False

    def _state_from(self, fields):
        if not any(f.startswith('SUBSYSTEM=power_supply') for f in fields):
            return None
        if 'POWER_SUPPLY_TYPE=Mains' in fields:
            online = next((f.partition('=')[2] for f in fields
                           if f.startswith('POWER_SUPPLY_ONLINE=')), None)
            if online is not None:
                return online.strip() == '1'
        # A battery's own news, or a port's, or a mains supply that did not say:
        # ask what the state is now, which is as true for one as for another.
        return self.plugged_in()
