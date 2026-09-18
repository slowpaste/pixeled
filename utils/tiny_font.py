import unicodedata

from PIL import Image, ImageDraw

# Every glyph is 3x5 and advances 4, so a run of n characters is 4n-1 wide.
TINY_WIDTH, TINY_HEIGHT, TINY_ADVANCE = 3, 5, 4

tiny_font = {
    '0': ["111", "101", "101", "101", "111"],
    '1': ["010", "110", "010", "010", "111"],
    '2': ["111", "001", "111", "100", "111"],
    '3': ["111", "001", "111", "001", "111"],
    '4': ["101", "101", "111", "001", "001"],
    '5': ["111", "100", "111", "001", "111"],
    '6': ["111", "100", "111", "101", "111"],
    '7': ["111", "001", "001", "001", "001"],
    '8': ["111", "101", "111", "101", "111"],
    '9': ["111", "101", "111", "001", "111"],
    ':': ["000", "010", "000", "010", "000"],
    'A': ["110", "101", "111", "101", "101"],
    'B': ["110", "101", "110", "101", "110"],
    'C': ["111", "100", "100", "100", "111"],
    'D': ["110", "101", "101", "101", "110"],
    'E': ["111", "100", "110", "100", "111"],
    'F': ["111", "100", "110", "100", "100"],
    'G': ["111", "100", "101", "101", "111"],
    'H': ["101", "101", "111", "101", "101"],
    'I': ["111", "010", "010", "010", "111"],
    'J': ["111", "001", "001", "101", "111"],
    'K': ["101", "101", "110", "101", "101"],
    'L': ["100", "100", "100", "100", "111"],
    'M': ["101", "111", "111", "101", "101"],
    'N': ["111", "101", "101", "101", "101"],
    'O': ["111", "101", "101", "101", "111"],
    'P': ["110", "101", "111", "100", "100"],
    'Q': ["111", "101", "101", "111", "011"],
    'R': ["110", "101", "111", "110", "101"],
    'S': ["111", "100", "111", "001", "111"],
    'T': ["111", "010", "010", "010", "010"],
    'U': ["101", "101", "101", "101", "111"],
    'V': ["101", "101", "101", "101", "010"],
    'W': ["101", "101", "111", "111", "101"],
    'X': ["101", "101", "010", "101", "101"],
    'Y': ["101", "101", "010", "010", "010"],
    'Z': ["111", "001", "010", "100", "111"],
    '-': ["000", "000", "111", "000", "000"],
    '_': ["000", "000", "000", "000", "111"],
    # Punctuation, for track titles. The gauges never needed any of this - they
    # spell out things they choose themselves - but a title is someone else's
    # text, and dropping every mark in it to a blank turns "DON'T STOP" into
    # gaps. Anything still unknown after sanitize_tiny_text is a space, which
    # is the right answer for a character this font cannot draw at 3x5.
    '.': ["000", "000", "000", "000", "010"],
    ',': ["000", "000", "000", "010", "100"],
    "'": ["010", "010", "000", "000", "000"],
    '"': ["101", "101", "000", "000", "000"],
    '!': ["010", "010", "010", "000", "010"],
    '?': ["111", "001", "011", "000", "010"],
    '&': ["010", "101", "110", "101", "011"],
    '(': ["001", "010", "010", "010", "001"],
    ')': ["100", "010", "010", "010", "100"],
    '/': ["001", "001", "010", "100", "100"],
    '+': ["000", "010", "111", "010", "000"],
    '*': ["000", "101", "010", "101", "000"],
    '#': ["101", "111", "101", "111", "101"],
    '%': ["101", "001", "010", "100", "101"],
    '=': ["000", "111", "000", "111", "000"],
}

# Typographic characters that have a plain equivalent this font does draw.
# Unicode normalisation does not fold these - a curly quote is its own
# character, not a decorated apostrophe - so they need naming.
TINY_FOLD = {
    '‘': "'", '’': "'", '‚': ',', '‛': "'",
    '“': '"', '”': '"', '„': '"',
    '‐': '-', '‑': '-', '‒': '-', '–': '-',
    '—': '-', '―': '-', '−': '-',
    '…': '...', '·': '.', '•': '*', '×': 'X',
    ' ': ' ', '​': '', '﻿': '',
}


def sanitize_tiny_text(text):
    """`text` reduced to characters this font can draw.

    Accents are folded rather than dropped - NFKD splits them into a base
    letter and a combining mark, so discarding only the marks leaves BJORK
    rather than BJ RK. What survives that and still has no glyph becomes a
    space, and runs of spaces collapse, so an unreadable script degrades to
    one gap instead of a field of them.
    """
    if not text:
        return ''
    folded = ''.join(TINY_FOLD.get(c, c) for c in text)
    decomposed = unicodedata.normalize('NFKD', folded)
    out = []
    for char in decomposed.upper():
        if unicodedata.combining(char):
            continue
        out.append(char if char in tiny_font else ' ')
    return ' '.join(''.join(out).split())


def tiny_text_width(text):
    """Pixels `text` occupies. The last character carries no trailing gap."""
    return max(len(text) * TINY_ADVANCE - 1, 0)


def run_width(run):
    """Pixels a run on a scrolling line occupies: text, or a picture put on
    the line in its place, like the sound visualizer, which has a width."""
    return tiny_text_width(run) if isinstance(run, str) else run.width


def draw_run(image, run, x, y):
    if isinstance(run, str):
        draw_tiny_text(image, run, x, y)
    else:
        run.draw(image, x, y)


def draw_tiny_text(image, text, x, y):
    draw = ImageDraw.Draw(image)
    for char in text:
        char_data = tiny_font.get(char, ["000"] * 5)
        for row, line in enumerate(char_data):
            for col, pixel in enumerate(line):
                if pixel == '1':
                    draw.point((x + col, y + row), fill=255)
        x += 4  # Move to the next character position


def cut_at_unseen_word(pieces, width):
    """A scrolling line cut back to the words already on the panel.

    `pieces` is the line as it was drawn: (text, x) runs, left to right, where
    a run may also be a picture (see run_width), which counts as a word. The
    cut falls where the first word that has not yet entered from the right
    edge begins, so whatever is on the panel finishes scrolling past and the
    next word is where something else can take over. Runs that have gone off
    the left edge are dropped.

    Returns (kept, x, cut): the runs to keep drawing, the x the next word
    would have been drawn at, and the (run, character) it starts at - None if
    every word is already showing, in which case x is one space past the end
    of the line. x is never short of the right edge, so what takes over always
    enters from off the panel rather than appearing in place.
    """
    kept = []
    end = None
    for run, (text, x) in enumerate(pieces):
        if not isinstance(text, str):
            # A picture is one word: it has come on or it has not.
            if x >= width:
                return kept, x, (run, 0)
            if x + text.width >= 0:
                kept.append((text, x))
            end = x + text.width + TINY_ADVANCE + 1
            continue
        for i, char in enumerate(text):
            start = x + i * TINY_ADVANCE
            if (start >= width and char != ' '
                    and (i == 0 or text[i - 1] == ' ')):
                head = text[:i].rstrip()
                if head and x + tiny_text_width(head) >= 0:
                    kept.append((head, x))
                return kept, start, (run, i)
        if text and x + tiny_text_width(text) >= 0:
            kept.append((text, x))
        end = x + (len(text) + 1) * TINY_ADVANCE
    return kept, max(width, end if end is not None else width), None

