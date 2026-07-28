"""Structure-preserving read-side elements: the ``.ftsl -> Element tree`` direction (J3c).

Loom's emitters run one way by design.  An authoring element holds *Signals* and
:meth:`~loom.scene.Element.emit` **bakes** them at a clock, so the ``.ftsl`` on disk is
a static snapshot of one frame.  :class:`~loom.iso.Isosurface` is the clearest case: its
field function, ``freq``, ``rotation``, ``drift``, ``placement`` and ``threshold`` are
all flattened into a single ``function { expr "…" }`` string, so no reader — however
clever, and no matter how precise the grammar — can recover
``Isosurface(field="gyroid", freq=3, …)`` from it.  The information is simply not there.

So the read direction cannot be "the inverse of emit", and pretending otherwise would
mean a reader that quietly invents parameters.  What it *can* be is exact about what the
text says: :class:`Block` holds a block's kind, name, ordered statements and nested
children verbatim, and re-emits them.  That makes the useful property **round-trip
fidelity** — ``parse -> emit`` reproduces the source — rather than authoring-object
recovery, and it is what the motivating consumer (an editor that loads a scene,
manipulates it and writes it back) actually needs.  ``TODO.md`` already proposed exactly
this shape for one case, "a new lightweight ``MeshRef`` element that re-emits the same
block"; :class:`Block` is that idea applied once, generally, instead of once per kind.

Where a block *does* map onto an authoring class without loss, the reader still builds
the real thing (:class:`~loom.scene.Material`, :class:`~loom.scene.Sphere`,
:class:`~loom.scene.Texture`, :class:`~loom.scene.Light`, :class:`~loom.scene.Camera`,
:class:`~loom.record.Record`) — see :func:`loom.grammar.reader.parse_element`.
:class:`Block` is the fallback for the baked kinds, not a replacement.

Two properties are deliberate:

**Statements are an ordered list, not a dict.**  Duplicate keys are normal in ``.ftsl``,
not an error: a ``camera_curve`` *is* a sequence of ``point`` / ``roll_at`` / ``fov_at``
lines, and a mix material a sequence of ``layer`` lines.  Folding them into a mapping
would silently keep only the last one and drop the geometry.

**Line layout is remembered.**  Each statement records whether a newline preceded it, so
``sphere { center 0 0 0  radius 1 }`` re-emits on one line while a ``medium { … }`` keeps
its indented multi-line form.  Without that, round-tripping a scene would reflow the
whole file and every diff would be unreadable.  A statement sharing a line also records
how many spaces separated it from the one before, because loom's own emitters are not
uniform — ``camera_curve`` writes ``up 0 1 0   fov_y 40   mode R`` with three spaces
where a material writes two — and a reader that normalised the gap would rewrite lines
it did not otherwise touch.

:class:`Document` finishes the job at file scope.  A :class:`Block` can only remember the
layout *inside* itself, but a file is also the blank lines that group its elements and the
top-level comments that head its sections — text that belongs to **neither** neighbouring
element and would be dropped by any element-list round-trip.  A ``Document`` is the
elements plus those literal gaps, so the whole file re-emits byte for byte.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from .scene import Element, EmitCtx

INDENT = 4          # spaces per nesting level, matching loom's own emitters
GAP = 2             # default spaces between two statements sharing a line


def _unquote(s: str) -> str:
    return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s


class Stmt:
    """One ``key value…`` statement inside a :class:`Block`.

    ``words`` are the verbatim source tokens, quotes included, so a value re-emits
    exactly as authored.  A statement may have **no** words: ``closed`` and
    ``exposure_lock`` are real valueless keywords, not malformed properties.

    ``raw`` is the statement's verbatim source slice, kept because whitespace *inside*
    a statement carries authorial intent that the token list cannot: hand-written
    scenes column-align their values (``units    meters`` under ``spectral 360 830 1``),
    and re-emitting ``units meters`` would silently reflow lines an editor never
    touched.  Any mutation drops it, so a changed statement renders from its words.
    """

    __slots__ = ("key", "words", "own_line", "gap", "raw", "before", "trail")

    def __init__(self, key: str, words: Sequence[str] = (), *, own_line: bool = True,
                 gap: int = GAP, raw: Optional[str] = None,
                 before: Sequence[str] = (), trail: str = "") -> None:
        self.key = str(key)
        self.words = [str(w) for w in words]
        self.own_line = bool(own_line)
        self.gap = max(1, int(gap))     # spaces before it when sharing a line
        self.raw = raw
        self.before = list(before)      # verbatim comment / blank lines above it
        self.trail = str(trail)         # the rest of its line (a trailing `# …`)

    # ---- value access ----------------------------------------------------
    @property
    def value(self) -> str:
        """The words joined by single spaces (``''`` for a valueless keyword)."""
        return " ".join(self.words)

    def unquoted(self) -> List[str]:
        """The words with surrounding double quotes stripped, for reading a name/path."""
        return [_unquote(w) for w in self.words]

    def numbers(self) -> List[float]:
        """The words as floats — raises :class:`ValueError` if any word isn't numeric."""
        return [float(w) for w in self.words]

    def set_words(self, words: Sequence[str]) -> None:
        """Replace the words, dropping the verbatim slice (it no longer describes them)."""
        self.words = [str(w) for w in words]
        self.raw = None

    def text(self) -> str:
        if self.raw is not None:
            return self.raw
        return self.key if not self.words else f"{self.key} {self.value}"

    def __repr__(self) -> str:
        return f"Stmt({self.key!r}, {self.words!r})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, Stmt) and other.key == self.key
                and other.words == self.words)


class Block(Element):
    """A parsed ``.ftsl`` block: ``[name =] kind ["quoted"] { <statements/children> }``.

    Static by construction — it came from baked text, so :meth:`roots` is empty and
    :meth:`emit` ignores the clock.  Mutate it in place (:meth:`set`, :meth:`add`) and
    re-emit to write a scene back out.
    """

    def __init__(self, kind: str, entries: Sequence[Union[Stmt, "Block"]] = (), *,
                 name: Optional[str] = None, quoted_name: Optional[str] = None,
                 subtype: Optional[str] = None, indent: int = INDENT,
                 brace_gap: int = 1, pad: Sequence[int] = (1, 1),
                 before: Sequence[str] = (), trail: str = "",
                 tail_before: Sequence[str] = (),
                 own_line: bool = True, gap: int = GAP) -> None:
        self.kind = str(kind)
        self.name = name
        self.quoted_name = quoted_name
        self.subtype = subtype          # legacy bareword subtype: `light area { … }`
        self.entries: List[Union[Stmt, Block]] = list(entries)
        self.indent = max(1, int(indent))   # body indent relative to the header column
        # Spaces between the header and `{`.  Hand-written scenes pad here to line the
        # braces up down a column (`material "red"   {` over `material "green" {`).
        self.brace_gap = max(1, int(brace_gap))
        # Padding just inside the braces of a one-line body, for the same reason:
        # `quad { … material red   }` lines its closing brace up with its neighbours'.
        self.pad = (max(1, int(pad[0])), max(1, int(pad[1])))
        self.before = list(before)          # comment / blank lines above the header
        self.trail = str(trail)             # the rest of the line the block ends on
        self.tail_before = list(tail_before)   # trivia between the last entry and `}`
        self.own_line = bool(own_line)
        self.gap = max(1, int(gap))

    # ---- Element contract ------------------------------------------------
    def roots(self) -> List:
        # A Block holds text, never Signals. The base Element.roots() walks vars(self)
        # looking for signal sites and would just churn over strings.
        return []

    def emit(self, ctx: Optional[EmitCtx] = None) -> str:   # ctx unused: nothing to bake
        return self.render(0)

    # ---- rendering -------------------------------------------------------
    def header(self) -> str:
        h = f"{self.name} = " if self.name else ""
        h += self.kind
        if self.quoted_name is not None:
            h += f' "{self.quoted_name}"'
        if self.subtype is not None:
            h += f" {self.subtype}"
        return h

    def render(self, indent: int = 0) -> str:
        """Render at ``indent`` spaces.  Single-line when no statement claims its own
        line, which is how ``sphere { … }`` stays on one line and ``medium { … }``
        doesn't.  Comment / blank lines are re-emitted where they were written."""
        head = self.header() + " " * self.brace_gap
        lpad, rpad = " " * self.pad[0], " " * self.pad[1]
        if not self.entries:
            return f"{head}{{{lpad}}}"
        multiline = any(e.own_line for e in self.entries)
        if not multiline:
            return f"{head}{{{lpad}{self._join(self.entries, 0)}{rpad}}}"

        body = indent + self.indent
        pad = " " * body
        lines: List[str] = []
        cur: List[Union[Stmt, Block]] = []

        def flush():
            # `before` belongs to the entry that opens the line, `trail` to the one
            # that closes it — everything between is on the line itself.  The first
            # group shares the header's line when its opening entry didn't claim one
            # (`group { translate …` with the rest of the body indented below).
            if not lines:
                lines.append(f"{head}{{" if cur[0].own_line else
                             f"{head}{{{lpad}{self._join(cur, body)}{cur[-1].trail}")
                if not cur[0].own_line:
                    return
            lines.extend(cur[0].before)
            lines.append(pad + self._join(cur, body) + cur[-1].trail)

        for e in self.entries:
            if e.own_line and cur:
                flush()
                cur = []
            cur.append(e)
        if cur:
            flush()
        lines.extend(self.tail_before)
        lines.append(" " * indent + "}")
        return "\n".join(lines)

    def _join(self, group: Sequence[Union[Stmt, "Block"]], indent: int) -> str:
        """One line's worth of entries, each preceded by the gap it was written with.

        A child block that owns its line renders at the deeper indent, so its own
        closing brace lines up with *its* header; one sharing a line renders flat.
        """
        out = []
        for i, e in enumerate(group):
            text = self._entry_text(e, indent if e.own_line else 0)
            out.append(text if i == 0 else " " * e.gap + text)
        return "".join(out)

    @staticmethod
    def _entry_text(entry: Union[Stmt, "Block"], indent: int) -> str:
        return entry.text() if isinstance(entry, Stmt) else entry.render(indent)

    def __repr__(self) -> str:
        return (f"Block({self.kind!r}, name={self.name!r}, "
                f"{len(self.entries)} entries)")

    # ---- inspection ------------------------------------------------------
    def stmts(self, key: Optional[str] = None) -> List[Stmt]:
        """Statements, optionally only those with ``key`` (in source order)."""
        return [e for e in self.entries
                if isinstance(e, Stmt) and (key is None or e.key == key)]

    def blocks(self, kind: Optional[str] = None) -> List["Block"]:
        """Child blocks, optionally only those of ``kind`` (in source order)."""
        return [e for e in self.entries
                if isinstance(e, Block) and (kind is None or e.kind == kind)]

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """The first ``key`` statement's value, or ``default`` when absent.

        Returns ``''`` for a *present but valueless* keyword, which is why the absence
        of a key and the presence of a bare one are distinguishable.
        """
        for s in self.stmts(key):
            return s.value
        return default

    def has(self, key: str) -> bool:
        return bool(self.stmts(key))

    def find(self, kind: str) -> Optional["Block"]:
        """The first child block of ``kind``, or ``None``."""
        for b in self.blocks(kind):
            return b
        return None

    def walk(self):
        """Yield this block and every descendant block, depth-first."""
        yield self
        for b in self.blocks():
            yield from b.walk()

    # ---- mutation --------------------------------------------------------
    def set(self, key: str, value: str = "") -> "Stmt":
        """Set ``key``'s value, replacing the first existing statement or appending a
        new one at the end.  The value is split on whitespace into words."""
        words = str(value).split()
        for s in self.stmts(key):
            s.set_words(words)
            return s
        s = Stmt(key, words)
        self.entries.append(s)
        return s

    def add(self, entry: Union[Stmt, "Block"]) -> Union[Stmt, "Block"]:
        """Append a statement or child block."""
        self.entries.append(entry)
        return entry

    def remove(self, key: str) -> int:
        """Drop every statement with ``key``; returns how many went."""
        before = len(self.entries)
        self.entries = [e for e in self.entries
                        if not (isinstance(e, Stmt) and e.key == key)]
        return before - len(self.entries)

    # ---- comparison ------------------------------------------------------
    def same_as(self, other: object) -> bool:
        """Structural equality **ignoring line layout** — two blocks that say the same
        thing formatted differently are ``same_as`` but not byte-identical."""
        if not isinstance(other, Block):
            return False
        if (self.kind, self.name, self.quoted_name, self.subtype) != \
           (other.kind, other.name, other.quoted_name, other.subtype):
            return False
        if len(self.entries) != len(other.entries):
            return False
        for a, b in zip(self.entries, other.entries):
            if isinstance(a, Stmt) != isinstance(b, Stmt):
                return False
            if isinstance(a, Stmt):
                if a != b:
                    return False
            elif not a.same_as(b):
                return False
        return True


class Document:
    """A whole ``.ftsl`` file: its elements **plus the literal text around them**.

    :class:`Block` recovers the layout *inside* an element, which is not enough on its
    own — a file is also the blank lines that group its elements and the top-level
    comments that head its sections, and those live **between** elements where no
    element can hold them.  A ``Document`` keeps them verbatim in :attr:`gaps`, so
    ``parse_document(text).emit(ctx) == text`` for a file loom can read, and an editor
    that rewrites one element leaves the file's shape alone.

    :attr:`gaps` always has one more entry than :attr:`elements`: ``gaps[0]`` is the text
    before the first element, ``gaps[i]`` the text between elements ``i-1`` and ``i``, and
    ``gaps[-1]`` the text after the last.
    """

    __slots__ = ("elements", "gaps")

    def __init__(self, elements: Sequence = (), gaps: Optional[Sequence[str]] = None):
        self.elements = list(elements)
        if gaps is None:
            gaps = [""] + ["\n\n"] * (len(self.elements) - 1) + ["\n"] \
                if self.elements else [""]
        self.gaps = list(gaps)
        if len(self.gaps) != len(self.elements) + 1:
            raise ValueError(f"a {len(self.elements)}-element Document needs "
                             f"{len(self.elements) + 1} gaps, got {len(self.gaps)}")

    # ---- sequence view ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, i):
        return self.elements[i]

    def blocks(self, kind: Optional[str] = None) -> List[Block]:
        """Every top-level :class:`Block` (optionally of one ``kind``)."""
        return [e for e in self.elements
                if isinstance(e, Block) and (kind is None or e.kind == kind)]

    # ---- mutation --------------------------------------------------------
    def insert(self, i: int, element, gap: str = "\n\n"):
        """Insert ``element`` at index ``i``, separated from its neighbour by ``gap``.

        The new gap goes *after* the element, except when appending, where it goes
        *before* — so a file's leading header text and its trailing newline both stay
        where they were instead of being pushed around the new element.
        """
        n = len(self.elements)
        i = min(max(i if i >= 0 else i + n, 0), n)
        self.elements.insert(i, element)
        self.gaps.insert(i if i == n else i + 1, gap)
        return element

    def append(self, element, gap: str = "\n\n"):
        return self.insert(len(self.elements), element, gap)

    def pop(self, i: int = -1):
        """Drop the element at ``i`` and one of its two gaps, keeping the file's leading
        and trailing text: the separator *after* it normally, the one *before* it when
        it is the last element."""
        n = len(self.elements)
        i = i if i >= 0 else i + n
        if not 0 <= i < n:
            raise IndexError("Document index out of range")
        el = self.elements.pop(i)
        del self.gaps[i if i == n - 1 else i + 1]
        return el

    # ---- emit ------------------------------------------------------------
    def emit(self, ctx: Optional[EmitCtx] = None) -> str:
        out = [self.gaps[0]]
        for i, el in enumerate(self.elements):
            out.append(el.emit(ctx))
            out.append(self.gaps[i + 1])
        return "".join(out)
