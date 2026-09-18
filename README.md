# pixeled

Drives the Framework 16's LED matrix input module as a small always-on display:
a clock, CPU and RAM meters, a battery gauge drawn as falling sand, a workspace
strip, the GNOME overview mirrored in outlines, album art for whatever is
playing, a sound visualizer, and ripples that cross the panel when the charger
or the headphones are plugged in.

The panel is 9x34 pixels with 256 brightness steps per pixel, refreshed about
60 times a second - which takes a patched firmware; see `firmware/README.md`.

## What it needs

- A Framework 16 with an LED matrix input module
- Fedora with GNOME on Wayland, as it ships
- `pulseaudio-utils` for `pactl` and `parec`; `setup.sh` installs it

## Setup

    git clone https://github.com/slowpaste/pixeled
    cd pixeled
    ./setup.sh

Run it as yourself, not with sudo; it asks for root only where it needs it.
It copies the code to `/opt/pixeled`, builds a venv there, installs the GNOME
extension, writes a `pixeled.service` that runs as you, and offers to flash the
firmware. Re-running it is the update path, and keeps the state the panel has
learned. `./setup.sh --help` lists what can be skipped.

The service is a system unit with no session bus, so the GNOME extension is how
it learns the workspace, the overview and the playing track's cover art. It also
puts the panel's brightness in quick settings, under the screen's own slider.

## The layout

`config.json` is a list of modules, top to bottom, each with a `position` (the
row it starts on) and a `height`. The one that ships shows the sound visualizer,
CPU and RAM, the clock, the workspace strip and the sand battery.

The visualizer row is `TransitIncidentsModule`, which announces WMATA bus
incidents and is the visualizer the rest of the time. Without an API key in
`~/.config/pixeled/wmata-key` it never has anything to announce, so it is simply
the visualizer - which is what most people want. `setup.sh` asks for a key, and
skipping it is fine.

`config.json.beamng` holds the settings for the driving dashboard, which takes
the whole panel while BeamNG has the keyboard - not merely while it is running,
since a game behind a browser is not being driven. It is fed by OutGauge over
UDP on localhost:8888. Switching OutSim on as
well, on 4444, lets the dash lean into corners: OutGauge carries no lateral
acceleration, and OutSim's yaw rate is where that comes from. Both switches are
in Options > Other > Protocols.

## What covers what

Four things can have the panel, and each slides in over the one below rather
than cutting to it, taking the top row at a word boundary so words already
going past finish:

    the overview      over everything, while it is open
    the dashboard     while BeamNG has the keyboard
    the artwork       while something is playing: the cover, and the visualizer
    the gauges        config.json, the rest of the time

Over all of them are the ripples from the headphone jack and the charger.

## Commands

    pixeled-speed          show the scroll rate for text
    pixeled-speed 32       set 32 px/s; also up, down and reset

## Branches

`main` is the general version. `personal` is the author's own, which differs
only in keeping the extension UUID their shell already has installed.
