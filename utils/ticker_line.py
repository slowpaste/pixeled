import math

from utils.tiny_font import cut_at_unseen_word, draw_run, run_width


class TickerLine:
    """A line of text scrolling along the title row, as a whole.

    What the overview and the artwork share, and the transit ticker hands the
    row to: words carried over from whoever had the row before, and then
    whatever the owner puts on it - text, or a picture such as the sound
    visualizer, which scrolls like a word. A place u on the line is drawn at
    width + u - floor(scroll), so things put on the line keep their distance
    from each other whatever starts or stops.

    Whatever is said next never replaces what is showing. It is put on the
    line where the next unseen word would have gone, and the line hurries the
    words ahead of it off: it accelerates, and brakes in time to be back at
    reading speed as the new words enter - or at rest, for something meant to
    stop in place. Every change of speed is a ramp, never a step.

    The row changes hands at a word boundary both ways. Taking it, the words
    already on the panel carry on past and the owner's text comes in where
    the next word would have; giving it back, the words on the panel go with
    it and the next owner carries on after them.
    """

    CATCH_UP = 4.0                  # times reading speed, hurrying words off
    RAMP_TIME = 0.25                # seconds from reading speed up to that

    def __init__(self, width, text_speed):
        self.width = width
        self.text_speed = text_speed
        self.reset()

    def reset(self):
        self.scroll = 0.0               # px the line has moved
        self.speed = self.text_speed    # px/s it is moving at
        self.lead = []                  # (text, u) of words carried over
        self.drawn = []                 # (text, x) runs on the panel last frame

    def accel(self):
        """px/s², scaled with the reading speed so the ramps keep their time."""
        return (self.CATCH_UP - 1) * self.text_speed / self.RAMP_TIME

    def x(self, u):
        return self.width + u - math.floor(self.scroll)

    def carry(self, pieces, x):
        """Keep `pieces` on the line; returns the u to follow them at, which
        is where panel column x is now."""
        base = math.floor(self.scroll) - self.width
        self.lead = [(text, px + base) for text, px in pieces]
        self.drawn = list(pieces)
        return x + base

    def follow_on(self):
        """Where the next thing goes: after the words on the panel now."""
        pieces, x, _ = cut_at_unseen_word(self.drawn, self.width)
        return self.carry(pieces, x)

    def rest(self, run):
        """Put `run` on the line to come to rest at the left edge; returns
        the u it goes at. The scroll that rests it is u + width.

        If it is already the last thing carried on the line, and not yet past
        its resting place, it stays where it is: the visualizer handed from one
        owner of the row to the next is the same one, and should not scroll
        off only to scroll back on. Otherwise it follows the words on the panel.
        """
        if self.lead and self.lead[-1][0] is run and self.x(self.lead[-1][1]) >= 0:
            return self.lead.pop()[1]
        return self.follow_on()

    def take(self, ticker):
        """Carry on from another ticker's words; returns the u to follow them
        at. It moves on at the speed they were going."""
        pieces, x, speed = ticker.hand_over(self.width)
        start = self.carry(pieces, x)
        self.speed = speed
        return start

    def give(self, dt):
        """The words on the panel, for another ticker to carry on from.

        Returns the keyword arguments to its take_back, and where the line was
        cut: the (run, character) of the first unseen word, or None if every
        word had come on.
        """
        pieces, x, cut = cut_at_unseen_word(self.drawn, self.width)
        return dict(pieces=pieces, x=x, width=self.width, speed=self.speed,
                    accel=self.accel(), dt=dt), cut

    def take_back(self, pieces, x, speed, dt):
        """Resume after words handed back; returns the u to follow them at.

        The words arrive moving at `speed`; move() ramps from there.
        """
        start = self.carry(pieces, x)
        self.speed = float(speed)
        return start

    def move(self, dt, goal, end):
        """Scroll, ramping toward the speed the line needs to be at.

        The goal is the scroll at which the next thing is where it should be,
        wanted at speed `end`. Short of it, the line goes as fast as it can
        still brake from in the distance left, up to CATCH_UP times reading
        speed, so hurrying and braking both fall out of the one rule. An `end`
        of 0 means coming to rest exactly at the goal.
        """
        dt = min(dt, 0.25)
        accel = self.accel()
        left = goal - self.scroll
        target = end
        if left > 0:
            # The fastest speed that can still brake to `end` by the goal,
            # counting the distance this frame covers on the way there -
            # without that the line starts braking a frame late and has to
            # stop short from speed.
            half = accel * dt / 2
            reach = max(0.0, 2 * accel * (left - self.speed * dt / 2))
            target = min(self.CATCH_UP * self.text_speed,
                         max(end, math.sqrt(half * half + end * end + reach) - half))
        step = accel * dt
        speed = max(self.speed - step, min(self.speed + step, target))
        self.scroll += (self.speed + speed) / 2 * dt
        self.speed = speed
        if end == 0.0 and self.scroll >= goal:
            # At rest. Braking lands within a frame of it, so the last
            # fraction of a pixel is simply dropped.
            self.scroll, self.speed = float(goal), 0.0

    def draw(self, image, placed, y=0):
        """Draw the carried words and then `placed`, (text, u) pairs."""
        self.lead = [(text, u) for text, u in self.lead
                     if self.x(u) + run_width(text) >= 0]
        runs = [(text, self.x(u)) for text, u in self.lead + list(placed)]
        for text, x in runs:
            draw_run(image, text, x, y)
        self.drawn = runs
