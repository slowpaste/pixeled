import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Meta from 'gi://Meta';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import {QuickSlider} from 'resource:///org/gnome/shell/ui/quickSettings.js';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

// ~/.cache rather than $XDG_RUNTIME_DIR: pixeled.service runs as a system unit
// with User=ecca, so it gets no XDG_RUNTIME_DIR of its own, and /run/user/1000
// does not exist yet if the service starts at boot before anyone logs in.
const STATE_DIR = GLib.build_filenamev([GLib.get_user_cache_dir(), 'pixeled']);
const STATE_FILE = GLib.build_filenamev([STATE_DIR, 'workspaces']);
const ARTWORK_FILE = GLib.build_filenamev([STATE_DIR, 'artwork']);
const OVERVIEW_FILE = GLib.build_filenamev([STATE_DIR, 'overview']);
// The panel's brightness, 0 to 1, as the quick settings slider has it. A
// preference rather than shell state, so it lives in ~/.config and survives the
// cache being cleared; pixeled.service reads it from the same home directory.
const BRIGHTNESS_FILE = GLib.build_filenamev(
    [GLib.get_user_config_dir(), 'pixeled', 'brightness']);
// ms. Dragging the slider changes its value every frame; the panel polls at
// 10Hz, so a write every 50ms is already more than it can see.
const BRIGHTNESS_COALESCE_MS = 50;
// ms between looks for the shell's brightness slider, which the panel adds to
// quick settings asynchronously and may not have yet when this is enabled.
const SLIDER_PLACE_RETRY_MS = 250;
const SLIDER_PLACE_ATTEMPTS = 40;

// ms. The swipe position arrives every frame the shell draws, in the overview
// and out of it; the panel polls at 30Hz and eases between what it reads, so
// writing faster buys nothing.
const POSITION_COALESCE_MS = 33;
// ms. Everything but the swipe - windows dragged between workspaces, titles
// changing, windows opening - is picked up on this heartbeat while the
// overview is open, rather than by following a signal on every window. An
// unchanged snapshot is not rewritten, so a still overview costs a JSON build.
const OVERVIEW_HEARTBEAT_MS = 250;

// s. MPRIS players do not announce Position as it advances, only when it
// jumps (Seeked), so while something plays it is re-read this often, to catch
// drift and players that seek without saying.
const POSITION_RESYNC_S = 5;

const MPRIS_NAMESPACE = 'org.mpris.MediaPlayer2';
const MPRIS_PATH = '/org/mpris/MediaPlayer2';
const MPRIS_PLAYER_IFACE = 'org.mpris.MediaPlayer2.Player';

// A quick settings slider for the LED matrix, placed directly under the
// screen brightness slider it sits beside in meaning. The icon is a grid of
// dots, which is what the panel is.
const PanelBrightnessSlider = GObject.registerClass({
    // Namespaced: GType names are global to the shell process.
    GTypeName: 'PixeledPanelBrightnessSlider',
}, class PanelBrightnessSlider extends QuickSlider {
    _init() {
        super._init({
            iconName: 'view-app-grid-symbolic',
            iconLabel: 'LED matrix brightness',
        });
        this.slider.accessible_name = 'LED matrix brightness';
    }
});

export default class PixeledWorkspacesExtension extends Extension {
    enable() {
        this._manager = global.workspace_manager;
        this._signalIds = [
            this._manager.connect('active-workspace-changed', () => this._publishAll()),
            this._manager.connect('notify::n-workspaces', () => this._publishAll()),
        ];

        // The overview gets a picture of its own on the panel: every workspace
        // as window outlines, and where the view is between them.
        //
        // Whether it is open is tracked from the signals, which are the stable
        // part. How far open it is has no public API: it is the controls'
        // state adjustment, 0 hidden, 1 the window picker, 2 the app grid,
        // which follows the finger through a swipe and eases through the
        // Super key's animation alike. That is private and could move between
        // shell versions, so it is looked up defensively, and without it the
        // panel gets no progress and falls back to a timed slide of its own.
        //
        // With progress the panel follows the overview all the way closed, so
        // it lets go on 'hidden'. Without, on 'hiding', so it starts its own
        // slide as the screen starts to close, not a quarter second after.
        this._overviewVisible = false;
        this._stateAdjustment = Main.overview._overview?.controls?._stateAdjustment ?? null;
        this._stateAdjustmentId = this._stateAdjustment?.connect('notify::value', () => {
            if (this._overviewVisible)
                this._scheduleOverview();
        }) ?? 0;
        this._overviewIds = [
            Main.overview.connect('showing', () => this._setOverviewVisible(true)),
            Main.overview.connect(this._stateAdjustment ? 'hidden' : 'hiding',
                () => this._setOverviewVisible(false)),
        ];

        // A private copy of the shell's shared workspace adjustment. Its value
        // is the view's position in workspaces - fractional while a swipe or a
        // switch animation is under way - which is what lets the panel follow
        // a finger rather than jump when the gesture ends. Bound
        // bidirectionally, so it is only ever read here.
        this._adjustment = Main.createWorkspacesAdjustment(Main.layoutManager.uiGroup);
        this._adjustmentId = this._adjustment.connect('notify::value', () => {
            this._scheduleWorkspaces();
            if (this._overviewVisible)
                this._scheduleOverview();
        });
        this._workspacesLast = null;
        this._workspacesPending = 0;
        this._overviewLast = null;
        this._overviewPending = 0;
        this._overviewHeartbeat = 0;
        this._hovered = null;
        this._motionId = 0;

        this._players = new Map();
        this._watchPlayers();
        this._positionTimer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT,
            POSITION_RESYNC_S, () => {
                for (const [, player] of this._players) {
                    if (player.proxy && player.status === 'Playing')
                        this._queryPosition(player);
                }
                return GLib.SOURCE_CONTINUE;
            });

        this._publishAll();  // the panel should be right from the moment we load
        this._publishArtwork();

        this._brightnessPending = 0;
        this._slider = null;
        this._placeSliderAttempts = 0;
        this._placeSliderTimer = 0;
        this._placeBrightnessSlider();
    }

    disable() {
        if (this._placeSliderTimer)
            GLib.source_remove(this._placeSliderTimer);
        this._placeSliderTimer = 0;
        if (this._brightnessPending) {
            GLib.source_remove(this._brightnessPending);
            this._brightnessPending = 0;
            this._writeBrightness();   // don't lose the last drag
        }
        if (this._slider) {
            // The quick settings menu parented the slider's (unused) popup
            // menu separately, so it has to be destroyed on its own.
            this._slider.menu?.destroy();
            this._slider.destroy();
            this._slider = null;
        }

        for (const id of this._signalIds ?? [])
            this._manager.disconnect(id);
        this._signalIds = null;
        this._manager = null;

        for (const id of this._overviewIds ?? [])
            Main.overview.disconnect(id);
        this._overviewIds = null;

        this._adjustment?.disconnect(this._adjustmentId);
        this._adjustment = null;
        if (this._stateAdjustmentId)
            this._stateAdjustment.disconnect(this._stateAdjustmentId);
        this._stateAdjustment = null;
        this._stateAdjustmentId = 0;
        this._stopOverviewTimers();
        this._stopHoverTracking();
        if (this._workspacesPending)
            GLib.source_remove(this._workspacesPending);
        this._workspacesPending = 0;

        if (this._nameOwnerId) {
            Gio.DBus.session.signal_unsubscribe(this._nameOwnerId);
            this._nameOwnerId = null;
        }
        if (this._positionTimer)
            GLib.source_remove(this._positionTimer);
        this._positionTimer = 0;
        for (const [, player] of this._players ?? []) {
            if (player.proxy && player.changedId)
                player.proxy.disconnect(player.changedId);
            if (player.proxy && player.seekedId)
                player.proxy.disconnect(player.seekedId);
        }
        this._players = null;

        // Leave nothing behind that would keep the panel showing artwork or
        // the overview over its gauges once we are gone.
        this._write(ARTWORK_FILE, JSON.stringify({}));
        this._write(OVERVIEW_FILE, JSON.stringify({visible: false}));
        this._overviewLast = null;
    }

    _publishAll() {
        this._publishWorkspaces();
        this._publishOverview();
    }

    _publishWorkspaces() {
        const count = this._manager.get_n_workspaces();
        const active = this._manager.get_active_workspace_index();

        // Dynamic workspaces keep one empty workspace past the last one in use,
        // so the raw count is always one higher than the number of workspaces
        // that hold anything. Whether to draw that spare is the reader's call,
        // so report the fact rather than deciding here.
        //
        // Recomputed on the two signals above, which is enough in practice:
        // putting the first window on the trailing workspace makes the shell
        // append a new empty one, and that changes n-workspaces.
        const last = this._manager.get_workspace_by_index(count - 1);
        const trailingEmpty = last !== null && last.list_windows().length === 0;

        // Where the view is between workspaces, fractional while a switch is
        // animating. The same shared adjustment the overview reads: outside
        // the overview the shell's workspace switch animation drives it too,
        // for a swipe and for a keyboard switch alike, so the indicator can
        // slide with the screen rather than jump when the switch lands.
        const position = Math.round((this._adjustment?.value ?? active) * 1000) / 1000;

        const payload = JSON.stringify({active, count, position, trailing_empty: trailingEmpty});
        if (payload === this._workspacesLast)
            return;
        this._workspacesLast = payload;
        this._write(STATE_FILE, payload);
    }

    _scheduleWorkspaces() {
        if (this._workspacesPending)
            return;
        this._workspacesPending = GLib.timeout_add(GLib.PRIORITY_DEFAULT,
            POSITION_COALESCE_MS, () => {
                this._workspacesPending = 0;
                this._publishWorkspaces();
                return GLib.SOURCE_REMOVE;
            });
    }

    // ── Overview ─────────────────────────────────────────────────────────────
    // What the overview shows, reduced to what a 9px-wide panel can draw of it:
    // for each workspace, the outline of every window the overview would show
    // there, and which of them was used last.

    _setOverviewVisible(visible) {
        this._overviewVisible = visible;
        this._stopOverviewTimers();
        this._stopHoverTracking();
        if (visible) {
            this._startHoverTracking();
            this._overviewHeartbeat = GLib.timeout_add(GLib.PRIORITY_DEFAULT,
                OVERVIEW_HEARTBEAT_MS, () => {
                    this._publishOverview();
                    return GLib.SOURCE_CONTINUE;
                });
        }
        this._publishOverview();
    }

    _scheduleOverview() {
        if (this._overviewPending)
            return;
        this._overviewPending = GLib.timeout_add(GLib.PRIORITY_DEFAULT,
            POSITION_COALESCE_MS, () => {
                this._overviewPending = 0;
                this._publishOverview();
                return GLib.SOURCE_REMOVE;
            });
    }

    // Which window the pointer last moved onto, for the panel to name. Only
    // pointer motion counts: windows sliding under a pointer left resting
    // during a workspace switch are not something anyone chose to point at.
    // Seen at the stage in the capture phase, ahead of the overview's own
    // handling, and always let through.
    _startHoverTracking() {
        this._hovered = null;
        this._motionId = global.stage.connect('captured-event', (stage, event) => {
            if (event.type() !== Clutter.EventType.MOTION)
                return Clutter.EVENT_PROPAGATE;
            try {
                const hovered = this._windowUnder(event);
                if (hovered !== this._hovered) {
                    this._hovered = hovered;
                    this._scheduleOverview();
                }
            } catch (e) {
                console.error(`pixeled: could not find the hovered window: ${e}`);
            }
            return Clutter.EVENT_PROPAGATE;
        });
    }

    _stopHoverTracking() {
        if (this._motionId)
            global.stage.disconnect(this._motionId);
        this._motionId = 0;
        this._hovered = null;
    }

    _windowUnder(event) {
        const [x, y] = event.get_coords();
        // The overview's window previews, and the clones in its workspace
        // thumbnails, carry the window they show as `metaWindow`. Whatever is
        // under the pointer - a preview's title or close button, say - is
        // walked up from until one of them is found.
        let actor = global.stage.get_actor_at_pos(Clutter.PickMode.REACTIVE, x, y);
        for (; actor; actor = actor.get_parent()) {
            if (actor.metaWindow)
                return actor.metaWindow.get_id();
        }
        return null;
    }

    _stopOverviewTimers() {
        if (this._overviewPending)
            GLib.source_remove(this._overviewPending);
        if (this._overviewHeartbeat)
            GLib.source_remove(this._overviewHeartbeat);
        this._overviewPending = 0;
        this._overviewHeartbeat = 0;
    }

    _publishOverview() {
        if (!this._overviewVisible) {
            this._writeOverview({visible: false});
            return;
        }

        const monitorIndex = Main.layoutManager.primaryIndex;
        const monitor = global.display.get_monitor_geometry(monitorIndex);
        const count = this._manager.get_n_workspaces();
        const workspaces = [];
        for (let i = 0; i < count; i++) {
            const workspace = this._manager.get_workspace_by_index(i);
            workspaces.push(this._workspaceWindows(workspace, monitorIndex, monitor));
        }

        this._writeOverview({
            visible: true,
            active: this._manager.get_active_workspace_index(),
            position: this._adjustment?.value ?? this._manager.get_active_workspace_index(),
            // How far the overview has come up, 0 to 1. The app grid lies
            // beyond 1 and still has the panel's overview fully up.
            ...this._stateAdjustment
                ? {progress: Math.max(0, Math.min(this._stateAdjustment.value, 1))}
                : {},
            // Height over width, so the panel can size a workspace to the
            // screen's shape rather than assuming one.
            aspect: monitor.width > 0 ? monitor.height / monitor.width : 0.625,
            // The id of the window the pointer last moved onto, or null.
            hovered: this._hovered,
            workspaces,
        });
    }

    _workspaceWindows(workspace, monitorIndex, monitor) {
        // The overview's own rule (Workspace._isMyWindow, _isOverviewWindow):
        // windows on this workspace and the primary monitor, less anything
        // that keeps itself out of the taskbar. The secondary monitors'
        // workspaces are not what the overview is swiping between.
        const windows = workspace.list_windows().filter(w =>
            w.located_on_workspace(workspace) &&
            w.get_monitor() === monitorIndex &&
            !w.skip_taskbar);

        // Most recently used first, which is the order the panel spells out
        // titles in: the window you were just in, then the one before.
        const recency = new Map(global.display
            .get_tab_list(Meta.TabList.NORMAL_ALL, workspace)
            .map((w, rank) => [w, rank]));

        // Bottom of the stack first, so the panel can paint them in order and
        // have the ones on top hide what they cover.
        return global.display.sort_windows_by_stacking(windows).map(w => {
            const rect = w.get_frame_rect();
            return {
                id: w.get_id(),
                x: (rect.x - monitor.x) / monitor.width,
                y: (rect.y - monitor.y) / monitor.height,
                w: rect.width / monitor.width,
                h: rect.height / monitor.height,
                rank: recency.get(w) ?? windows.length,
                title: w.get_title() ?? '',
            };
        });
    }

    _writeOverview(state) {
        const payload = JSON.stringify(state);
        if (payload === this._overviewLast)
            return;
        this._overviewLast = payload;
        this._write(OVERVIEW_FILE, payload);
    }

    // ── Artwork ──────────────────────────────────────────────────────────────
    // The current track's cover while something is playing, with its title and
    // artist so the panel can spell out which track it is, and when it will
    // end, so the panel can give a track's last moments to the sound.

    _publishArtwork() {
        const track = this._playingTrack();
        this._write(ARTWORK_FILE, JSON.stringify({
            art: track?.art ?? null,
            // The panel spells these out between passes over the cover, so
            // they travel with the art rather than being looked up separately:
            // a cover and the name of something else is worse than no name.
            title: track?.title ?? null,
            artist: track?.artist ?? null,
            // Seconds since the epoch at which the track will end at the rate
            // it is playing, or null where the player does not say how long
            // it is or where it has got to - a stream, most often.
            ends_at: track?.endsAt ?? null,
            playing: track !== null,
        }));
    }

    _playingTrack() {
        for (const [, player] of this._players) {
            if (player.status !== 'Playing' || !player.art)
                continue;
            let art = player.art;
            if (art.startsWith('file://')) {
                // http(s) is left as-is; the panel fetches and caches it.
                try {
                    [art] = GLib.filename_from_uri(art);
                } catch {
                    continue;
                }
            }
            return {art, title: player.title, artist: player.artist,
                    endsAt: this._endsAt(player)};
        }
        return null;
    }

    _endsAt(player) {
        if (player.length === null || player.position === null || !(player.rate > 0))
            return null;
        const left = (player.length - player.position) / player.rate;
        return (player.positionAt + left) / 1e6;
    }

    _queryPosition(entry) {
        entry.proxy.call('org.freedesktop.DBus.Properties.Get',
            new GLib.Variant('(ss)', [MPRIS_PLAYER_IFACE, 'Position']),
            Gio.DBusCallFlags.NONE, 1000, null, (proxy, res) => {
                try {
                    const [position] = proxy.call_finish(res).recursiveUnpack();
                    entry.position = Number(position);
                    entry.positionAt = GLib.get_real_time();
                } catch {
                    entry.position = null;   // a player that will not say
                }
                if (this._players)
                    this._publishArtwork();
            });
    }

    _watchPlayers() {
        // Players come and go, so track the bus names rather than probing.
        this._nameOwnerId = Gio.DBus.session.signal_subscribe(
            'org.freedesktop.DBus', 'org.freedesktop.DBus', 'NameOwnerChanged',
            '/org/freedesktop/DBus', MPRIS_NAMESPACE,
            Gio.DBusSignalFlags.MATCH_ARG0_NAMESPACE,
            (conn, sender, path, iface, signal, params) => {
                const [name, , newOwner] = params.deepUnpack();
                if (newOwner)
                    this._addPlayer(name);
                else if (this._players.delete(name))
                    this._publishArtwork();
            });

        Gio.DBus.session.call(
            'org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus',
            'ListNames', null, null, Gio.DBusCallFlags.NONE, -1, null,
            (conn, res) => {
                try {
                    const [names] = conn.call_finish(res).deepUnpack();
                    for (const name of names) {
                        if (name.startsWith(`${MPRIS_NAMESPACE}.`))
                            this._addPlayer(name);
                    }
                } catch (e) {
                    console.error(`pixeled: could not list MPRIS players: ${e}`);
                }
            });
    }

    _addPlayer(name) {
        if (this._players.has(name))
            return;
        const entry = {proxy: null, changedId: 0, seekedId: 0, status: null,
                       art: null, title: null, artist: null, track: null,
                       length: null, position: null, positionAt: 0, rate: 1};
        this._players.set(name, entry);

        Gio.DBusProxy.new(
            Gio.DBus.session, Gio.DBusProxyFlags.NONE, null,
            name, MPRIS_PATH, MPRIS_PLAYER_IFACE, null,
            (proxy_, res) => {
                let proxy;
                try {
                    proxy = Gio.DBusProxy.new_finish(res);
                } catch {
                    this._players?.delete(name);
                    return;
                }
                if (!this._players?.has(name))
                    return;   // disabled, or the player quit, while we waited
                entry.proxy = proxy;
                entry.changedId = proxy.connect('g-properties-changed',
                    () => this._refreshPlayer(entry));
                entry.seekedId = proxy.connect('g-signal', (p, sender, signal, params) => {
                    if (signal !== 'Seeked')
                        return;
                    entry.position = Number(params.recursiveUnpack()[0]);
                    entry.positionAt = GLib.get_real_time();
                    this._publishArtwork();
                });
                this._refreshPlayer(entry);
            });
    }

    _refreshPlayer(entry) {
        const proxy = entry.proxy;
        entry.status = proxy.get_cached_property('PlaybackStatus')?.deepUnpack() ?? null;

        const metadata = proxy.get_cached_property('Metadata')?.deepUnpack() ?? {};
        entry.art = metadata['mpris:artUrl']?.deepUnpack() ?? null;
        entry.title = metadata['xesam:title']?.deepUnpack() || null;

        // xesam:artist is an array - collaborations list every performer. The
        // panel has one line to spend, so join them and let it decide what
        // fits, rather than picking one here and calling it the artist.
        // albumArtist is the fallback compilations tend to fill in instead.
        const artists = metadata['xesam:artist']?.deepUnpack()
            ?? metadata['xesam:albumArtist']?.deepUnpack() ?? [];
        entry.artist = (Array.isArray(artists) ? artists : [artists])
            .filter(a => typeof a === 'string' && a.length).join(', ') || null;

        const length = Number(metadata['mpris:length']?.deepUnpack());
        entry.length = length > 0 ? length : null;
        entry.rate = proxy.get_cached_property('Rate')?.deepUnpack() ?? 1;

        // A new track's position is unknown until asked: the last one's would
        // put its end somewhere it is not.
        const track = JSON.stringify([metadata['mpris:trackid']?.deepUnpack() ?? null,
            entry.title, entry.artist, entry.art, entry.length]);
        if (track !== entry.track) {
            entry.track = track;
            entry.position = null;
        }

        this._publishArtwork();
        if (entry.status === 'Playing')
            this._queryPosition(entry);
    }

    // ── Panel brightness ─────────────────────────────────────────────────────

    _placeBrightnessSlider() {
        const quickSettings = Main.panel.statusArea.quickSettings;
        const brightness = quickSettings?._brightness?.quickSettingsItems?.[0];
        // The shell's own slider has a parent once the panel has put its items
        // into the menu. Placing ours before that would have them inserted
        // around it, and it would land wherever that left it.
        if (!brightness?.get_parent()) {
            if (++this._placeSliderAttempts > SLIDER_PLACE_ATTEMPTS) {
                console.error('pixeled: quick settings brightness slider never appeared');
                return;
            }
            this._placeSliderTimer = GLib.timeout_add(GLib.PRIORITY_DEFAULT,
                SLIDER_PLACE_RETRY_MS, () => {
                    this._placeSliderTimer = 0;
                    this._placeBrightnessSlider();
                    return GLib.SOURCE_REMOVE;
                });
            return;
        }

        this._slider = new PanelBrightnessSlider();
        this._slider.slider.value = this._readBrightness();
        this._slider.slider.connect('notify::value', () => this._scheduleBrightness());

        // Full width, like the shell's sliders, and directly under its
        // brightness slider.
        const menu = quickSettings.menu;
        const next = brightness.get_next_sibling();
        if (next)
            menu.insertItemBefore(this._slider, next, 2);
        else
            menu.addItem(this._slider, 2);
    }

    _readBrightness() {
        try {
            const [, contents] = GLib.file_get_contents(BRIGHTNESS_FILE);
            const value = parseFloat(new TextDecoder().decode(contents));
            if (Number.isFinite(value))
                return Math.max(0, Math.min(value, 1));
        } catch {
            // Never set: the panel starts at full brightness.
        }
        return 1;
    }

    _scheduleBrightness() {
        if (this._brightnessPending)
            return;
        this._brightnessPending = GLib.timeout_add(GLib.PRIORITY_DEFAULT,
            BRIGHTNESS_COALESCE_MS, () => {
                this._brightnessPending = 0;
                this._writeBrightness();
                return GLib.SOURCE_REMOVE;
            });
    }

    _writeBrightness() {
        if (this._slider)
            this._write(BRIGHTNESS_FILE, `${this._slider.slider.value.toFixed(3)}\n`);
    }

    _write(path, payload) {
        try {
            GLib.mkdir_with_parents(GLib.path_get_dirname(path), 0o755);
            // g_file_set_contents writes a temporary file and renames it over
            // the target, so a reader polling at 10Hz sees either the old
            // contents or the new ones, never a half-written file.
            GLib.file_set_contents(path, payload);
        } catch (e) {
            console.error(`pixeled: could not write ${path}: ${e}`);
        }
    }
}
