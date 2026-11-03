"""Turns sentences into speech while they are still being written, and hands the audio out in order.

Brain.answer_stream() writes a reply one sentence at a time. A SpeechSession starts making the audio of each
sentence as soon as it arrives (a few at once, because Microsoft's voice takes a second or two before it sends
anything) and gives the audio back in the order the sentences were added, so the speaker can start on the
first sentence while the later ones are still being made.

    session = tts.session()
    session.add("It's twenty five degrees in Rajkot.")     # its audio starts being made now
    session.add("Light rain later.")
    session.close()                                        # no more sentences are coming
    for tag, audio, first in session.pieces():             # MP3 (or PCM) bytes, in order
        ...

One voice says the whole reply, because MP3 and PCM can't be mixed in one reply. Voices are tried in order
(Edge, then the laptop's own) and the first sentence that gets audio decides; a later sentence that another
voice happened to make is made again with the chosen one. A sentence that can't be made ends the reply there
(what was said stays said), as it always did.

A voice that takes a sentence and then says nothing for `stall_s` (a connection that died quietly) is given up
on: if the reply has no voice yet, the sentence is made again by the next one; otherwise the reply ends there.

A voice is anything with .name, .provider, .fmt and speak(text, put): speak() calls put(audio bytes) for each
piece as it is made and raises if it fails. put() raises Stop when the sentence isn't wanted any more.
"""

from __future__ import annotations

import collections
import threading
import time

import config


class VoiceError(RuntimeError):
    """No voice could make a sentence."""


class Stop(Exception):
    """Raised by put() inside a voice's speak(): this sentence isn't wanted any more, stop making it."""


_REMAKE = object()


class _Item:
    """One sentence on its way to becoming audio."""

    __slots__ = ("text", "tag", "pieces", "gen", "started", "counted", "finished", "dropped", "error", "voice",
                 "skip", "current", "t_add", "t_first", "t_last")

    def __init__(self, text: str, tag) -> None:
        self.text = text
        self.tag = tag
        self.pieces: collections.deque = collections.deque()   # audio made but not handed out yet
        self.gen = 0               # bumped when the sentence is made again, so the old attempt is ignored
        self.started = False       # a thread is making it
        self.counted = False       # ... and it holds one of the parallel places
        self.finished = False      # that thread is done, with audio or an error
        self.dropped = False       # nobody wants it any more
        self.error = None
        self.voice = None          # the voice that made (or is making) it
        self.skip: list = []       # voices that went quiet on this sentence: not asked again
        self.current = None        # the voice being asked now
        self.t_add = time.monotonic()
        self.t_first = None        # when its first audio arrived
        self.t_last = None         # ... and the latest


class SpeechSession:
    """One reply's worth of speech. The writer side (add, close, drop_unsent, abort) and the reader side
    (peek, head_error, pieces) can be used from different threads; they share one lock, `cv`, which a
    caller may pass in so that it can wait for changes here and its own at the same time."""

    def __init__(self, voices, parallel: int | None = None, cond: threading.Condition | None = None,
                 stall_s: float | None = None) -> None:
        self._voices = list(voices)
        if not self._voices:
            raise ValueError("a speech session needs at least one voice")
        self._parallel = max(1, int(parallel or getattr(config, "TTS_PARALLEL", 3)))
        self._stall_s = float(stall_s if stall_s is not None else config.EDGE_TIMEOUT)
        self.cv = cond or threading.Condition()
        self.voice = None                 # the voice the reply is being said in, once the first piece goes out
        self.error: Exception | None = None   # why the audio stopped before the end of the reply, if it did
        self._items: list[_Item] = []
        self._pos = 0                     # the sentence being handed out
        self._head_taken = False          # ... and part of it has gone
        self._waiting: collections.deque = collections.deque()
        self._running = 0
        self._closed = False
        self._aborted = False

    # ------------------------------------------------------------ the writer's side

    def add(self, text: str, tag=None) -> bool:
        """Say this next. Starts making its audio now (or as soon as one of the parallel places is free)."""
        text = (text or "").strip()
        if not text:
            return False
        with self.cv:
            if self._closed or self._aborted:
                return False
            item = _Item(text, tag)
            self._items.append(item)
            self._waiting.append(item)
            self._pump()
            self.cv.notify_all()
        return True

    def close(self) -> None:
        """No more sentences are coming: pieces() ends once everything has been handed out."""
        with self.cv:
            self._closed = True
            self.cv.notify_all()

    def drop_unsent(self) -> int:
        """Forget the sentences that haven't started going out (the one being sent stays). Returns how many."""
        with self.cv:
            cut = self._pos + (1 if self._head_taken else 0)
            gone = self._items[cut:]
            del self._items[cut:]
            for item in gone:
                self._forget(item)
            self._waiting = collections.deque(i for i in self._waiting if not i.dropped)
            self.cv.notify_all()
            return len(gone)

    def abort(self) -> None:
        """Stop everything: the speaker has gone, or the turn failed. Wakes anyone waiting in pieces()."""
        with self.cv:
            self._aborted = self._closed = True
            for item in self._items[self._pos:]:
                self._forget(item)
            self.cv.notify_all()

    def _forget(self, item: _Item) -> None:
        item.dropped = True
        item.pieces.clear()
        if item.counted:                  # its thread stops at its next piece; don't hold a place meanwhile
            item.counted = False
            self._running -= 1

    # ------------------------------------------------------------ the reader's side

    def peek(self):
        """(voice, ms its first audio took) once the next sentence has audio ready to send, else None."""
        with self.cv:
            head = self._head()
            if head is None or self._head_taken or not head.pieces:
                return None
            return head.voice, int((head.t_first - head.t_add) * 1000)

    def head_error(self):
        """The error if the first sentence couldn't be made by any voice (so there's nothing to say), else None."""
        with self.cv:
            head = self._head()
            if head and head.finished and head.error is not None and not head.pieces and not self._head_taken:
                return head.error
            return None

    def is_empty(self) -> bool:
        with self.cv:
            return self._head() is None

    def poll(self) -> None:
        """For a reader that is waiting to see whether there is a reply at all (it isn't taking pieces yet): gives
        up on a voice that has gone quiet, the way pieces() does for itself once the reply is going out."""
        with self.cv:
            head = self._head()
            if (head is None or self._aborted or not head.started or head.finished or head.pieces
                    or self._head_taken or time.monotonic() - head.t_last < self._stall_s):
                return
            if not self._unstick(head):
                self._give_up(head)
            self.cv.notify_all()

    @property
    def aborted(self) -> bool:
        return self._aborted

    def pieces(self):
        """Yield (tag, audio bytes, first) for the whole reply in order, waiting for what isn't made yet;
        `first` is True for the first piece of each sentence. Ends after close() once all of it has been
        handed out, or early on abort() or when a sentence can't be made (see .error)."""
        while True:
            with self.cv:
                got = self._next()
            if got is None:
                return
            if got[0] is _REMAKE:
                if not self._remake(got[1]):
                    return
                continue
            yield got

    # ------------------------------------------------------------ inside

    def _head(self):
        return self._items[self._pos] if self._pos < len(self._items) else None

    def _next(self):
        """With the lock held: what to hand out next, waiting for it if need be. None = the reply is over."""
        while True:
            if self._aborted:
                return None
            head = self._head()
            if head is None:
                if self._closed:
                    return None
                self.cv.wait(1.0)
                continue
            if head.pieces:
                if self.voice is not None and head.voice is not self.voice:
                    return (_REMAKE, head)
                data = head.pieces.popleft()
                first = not self._head_taken
                self._head_taken = True
                if self.voice is None:
                    self.voice = head.voice
                return (head.tag, data, first)
            if head.finished:
                if head.error is not None:
                    self.error = head.error
                    return None
                self._pos += 1
                self._head_taken = False
                continue
            if head.started:
                quiet = time.monotonic() - head.t_last
                if quiet >= self._stall_s:
                    if self._unstick(head):
                        continue
                    self._give_up(head)
                    self.error = head.error
                    return None
                self.cv.wait(max(0.01, min(1.0, self._stall_s - quiet)))
            else:
                self.cv.wait(1.0)

    def _release(self, item: _Item) -> None:
        """With the lock held: the sentence gives its place back (its request, if still running, is ignored)."""
        if item.counted:
            item.counted = False
            self._running -= 1

    def _unstick(self, item: _Item) -> bool:
        """With the lock held: the voice making this sentence has gone quiet. If it hasn't made any audio yet
        and there is a voice left that the reply may still use, start the sentence again with that one."""
        if item.voice is not None or item.pieces or item.dropped or self._aborted:
            return False
        gone = item.skip + ([item.current] if item.current is not None else [])
        if not [v for v in (self._voices if self.voice is None else [self.voice]) if v not in gone]:
            return False
        print(f"[tts] {getattr(item.current, 'name', 'the voice')} has said nothing for {self._stall_s:g} s; "
              f"making the sentence with another voice")
        self._tell_stalled(item)
        item.skip = gone
        item.current = None
        item.gen += 1                          # whatever the old request still does is ignored
        self._release(item)
        item.started = item.finished = False
        item.error = None
        self._waiting.appendleft(item)
        self._pump()
        return True

    @staticmethod
    def _tell_stalled(item: _Item) -> None:
        """A voice that keeps track of its own health (Edge) is told that a request to it went unanswered."""
        tell = getattr(item.current, "stalled", None)
        if tell is not None:
            try:
                tell()
            except Exception:
                pass

    def _give_up(self, item: _Item) -> None:
        """With the lock held: this sentence is not going to be made."""
        self._tell_stalled(item)
        item.error = VoiceError("the voice stalled")
        item.finished = True
        item.dropped = True
        item.gen += 1
        self._release(item)
        self.cv.notify_all()

    def _pump(self) -> None:
        """With the lock held: start making waiting sentences while there are places free."""
        while self._waiting and self._running < self._parallel:
            item = self._waiting.popleft()
            if item.dropped:
                continue
            item.started = item.counted = True
            item.t_last = time.monotonic()
            self._running += 1
            threading.Thread(target=self._run, args=(item, item.gen), name="voice", daemon=True).start()

    def _run(self, item: _Item, gen: int) -> None:
        try:
            self._make(item, gen)
        except Stop:
            pass
        except Exception as exc:          # a bug in a voice must not leave the reply waiting for ever
            self._fail(item, gen, VoiceError(f"{type(exc).__name__}: {exc}"))
        finally:
            with self.cv:
                if item.gen == gen:               # not an old request that was given up on
                    item.finished = True
                    self._release(item)
                self._pump()
                self.cv.notify_all()

    def _make(self, item: _Item, gen: int) -> None:
        with self.cv:
            pool = [self.voice] if self.voice is not None else list(self._voices)
            chain = [v for v in pool if v not in item.skip]
        errors = []
        for voice in chain:
            with self.cv:
                if item.gen != gen or item.dropped or self._aborted:
                    return                        # given up on while an earlier voice was being asked
                item.current = voice
            name = getattr(voice, "name", "voice")
            try:
                voice.speak(item.text, lambda data, v=voice: self._put(item, gen, v, data))
            except Stop:
                return
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                if item.voice is not None:        # it had started: no other voice can carry on from the middle
                    break
                print(f"[tts] {name} failed ({exc}); trying the next voice")
                continue
            if item.voice is None:
                errors.append(f"{name}: no audio")
                continue
            return
        self._fail(item, gen, VoiceError("; ".join(errors) or "no voice left"))

    def _fail(self, item: _Item, gen: int, error: Exception) -> None:
        with self.cv:
            if item.gen == gen and not item.dropped:
                item.error = error
            self.cv.notify_all()

    def _put(self, item: _Item, gen: int, voice, data: bytes) -> None:
        if not data:
            return
        with self.cv:
            if self._aborted or item.dropped or item.gen != gen:
                raise Stop()
            now = time.monotonic()
            if item.voice is None:
                item.voice = voice
                item.t_first = now
            item.pieces.append(data)
            item.t_last = now
            self.cv.notify_all()

    def _remake(self, item: _Item) -> bool:
        """A sentence made by a different voice than the one the reply is in: make it again with that one."""
        with self.cv:
            item.gen += 1                     # whatever the old attempt still does is ignored
            item.pieces.clear()
            self._release(item)
            voice = self.voice
        parts: list[bytes] = []
        try:
            voice.speak(item.text, parts.append)
            if not any(parts):
                raise VoiceError("no audio")
        except Exception as exc:
            with self.cv:
                self.error = VoiceError(f"{getattr(voice, 'name', 'voice')}: {exc}")
            return False
        with self.cv:
            if self._aborted:
                return False
            item.pieces.extend(p for p in parts if p)
            item.voice = voice
            item.error = None
            item.finished = True
        return True
