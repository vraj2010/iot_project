"""The reply side of a streamed turn: speak each sentence as Gemini writes it, and send the audio on.

Brain.answer_stream() tells this class about every sentence (sentence()) and about each round of writing
that is over (round_end()). It starts the voice on each sentence (SpeechSession), decides when to begin the
HTTP reply, and a writer thread then sends the audio to the speaker in order while the rest is still being
made. The brain keeps running in the caller's thread; nothing here ever blocks it for long.

The frame header goes out first, together with the first audio, and can't be changed afterwards. It carries
the text and the music action, and the action is only known once the tools have run. So the reply is not
begun ("decided") before the round that is writing it has ended without a tool call, or has been going on
for `hold_s` (a long answer shouldn't keep the speaker waiting). From then on the music can't be changed in
this turn: music_ok() says so, and the brain tells Gemini.

While a slow tool runs (a web search, a weather lookup that isn't cached) there is nothing to say yet, so a
short line like "One sec, boss." goes out at once instead of silence. Not if the voice that line would come
from looks unreachable (filler_ok): it comes from the voice cache, which makes that voice the voice of the whole
reply, and a reply that can't go on after its first line is worse than a short silence.

The reply is sent through a `transport`: begin() starts the HTTP response, write(bytes) sends a piece of
the body, end() finishes it.
"""

from __future__ import annotations

import random
import threading
import time

FILLER = "filler"           # the tag of the line said while a tool runs

_last_filler: list = [None]


class StreamingReply:
    """The sink for Brain.answer_stream() that actually talks to the speaker."""

    def __init__(self, session, transport, build_header, get_actions, *, offset_ms: int = 0, hold_s: float = 1.5,
                 fillers=(), apology: str = "", filler_ok=None, clock=time.monotonic) -> None:
        """build_header(voice, text, chat_ms, tts_ms, actions) -> the frame header bytes.
        get_actions() -> the music action the speaker is to be given. offset_ms is how long the speaker
        has been waiting already (hearing it), so the times we report are from the end of its question."""
        self.session = session
        self.cv = session.cv                   # one lock for the voice and for us
        self._transport = transport
        self._build_header = build_header
        self._get_actions = get_actions
        self._offset_ms = offset_ms
        self._hold_s = hold_s
        self._fillers = list(fillers)
        self._apology = apology
        self._filler_ok = filler_ok            # () -> bool: is it all right to start with a line from the cache?
        self._clock = clock
        self._t0 = clock()

        self._spoken: list[str] = []           # sentences already on their way to the speaker
        self._draft: list[str] = []            # the round being written now
        self._t_draft = None                   # when its first sentence came
        self._round = 0                        # which round of Gemini's writing (a tool call ends one)
        self._round_audio: dict[int, int] = {}      # round -> ms until its first audio was sent
        self._final = False                    # the round that is the answer has ended
        self._finished = False                 # the brain is done (or has failed)
        self._failed = False                   # it failed before anything was sent
        self._decided = False                  # the reply has begun (or is about to)
        self._voice_dead = False               # no voice could make the first sentence
        self._stopped = False                  # the voice failed part way through the reply: it is over
        self._thread: threading.Thread | None = None

        self.dead = False                      # the speaker hung up
        self.started = False                   # the HTTP response has begun
        self.voice = None                      # the voice that is speaking
        self.voice_error: Exception | None = None
        self.filler: str | None = None
        self.tools_ran = False                 # a round ended in tool calls: they have run, or are about to
        self.first_audio_ms: int | None = None      # from the end of the question to the first audio sent
        self.chat_ms: int | None = None
        self.tts_ms: int | None = None              # how long the first sentence took to become audio
        self.audio = bytearray()                    # everything sent, to save as last_reply

    # ------------------------------------------------------------ what Brain.answer_stream() calls

    def sentence(self, text: str) -> None:
        with self.cv:
            if self.dead or self._voice_dead or self._stopped:
                return
            now = self._clock()
            if not self._draft:
                self._t_draft = now
            self._draft.append(text)
            self.session.add(text, tag=self._round)
            self.cv.notify_all()

    def round_end(self, final: bool, tools=(), slow: bool = False) -> None:
        with self.cv:
            if final:
                self._final = True
                self.cv.notify_all()
                return
            # A round that ended in a tool call isn't the answer: whatever it said before the call is dropped,
            # unless it has already started to go out.
            self.tools_ran = True
            if self._decided:
                self._spoken += self._draft
            else:
                self.session.drop_unsent()
            self._draft = []
            self._t_draft = None
            self._round += 1
            if (slow and not self._decided and self._fillers and not (self.dead or self._voice_dead)
                    and (self._filler_ok is None or self._filler_ok())):
                line = self._pick_filler()
                if self.session.add(line, tag=FILLER):
                    self._spoken.append(line)
                    self.filler = line
                    self._decided = True        # decided now, so that no music can slip in behind it
            self.cv.notify_all()

    def cancelled(self) -> bool:
        """The brain asks this between chunks: stop if nobody can hear the rest (the speaker hung up, or the voice
        failed part way through). A voice that never worked at all is not this: the caller still wants the text."""
        return self.dead or self._stopped

    @property
    def answer_ms(self) -> int | None:
        """From the end of the question to the first audio of the answer itself: that of the last round that was
        heard, so neither the filler nor something said before a tool was called."""
        rounds = dict(self._round_audio)
        return rounds[max(rounds)] if rounds else None

    def music_ok(self) -> bool:
        with self.cv:
            return not self._decided

    # ------------------------------------------------------------ what the caller does

    @property
    def decided(self) -> bool:
        return self._decided

    def start(self) -> None:
        """Start the writer. Call before the brain starts."""
        self._thread = threading.Thread(target=self._write, name="reply-writer", daemon=True)
        self._thread.start()

    def finish(self, error: BaseException | None = None) -> None:
        """The brain is done, or has failed. After this, no more sentences come."""
        with self.cv:
            self._finished = True
            if error is None:
                self._final = True
                self.session.close()
            elif not self._decided:
                self._failed = True             # nothing has been sent: the caller can still answer with an error
                self.session.abort()
            else:
                if self._apology and not (self.dead or self._voice_dead or self._stopped):
                    self.session.add(self._apology)     # don't leave the speaker with a sentence cut short
                self.session.close()
            self.cv.notify_all()

    def wait(self, timeout: float | None = None) -> None:
        """Wait until everything has been sent."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def close(self) -> None:
        """Always call this when the turn is over, however it ended: stops whatever is still being made."""
        self.session.abort()
        self.wait(5)

    def text(self) -> str:
        with self.cv:
            return " ".join(self._spoken + self._draft)

    # ------------------------------------------------------------ the writer

    def _pick_filler(self) -> str:
        options = [f for f in self._fillers if f != _last_filler[0]] or self._fillers
        line = random.choice(options)
        _last_filler[0] = line
        return line

    def _since_start_ms(self) -> int:
        return self._offset_ms + int((self._clock() - self._t0) * 1000)

    def _gate_open(self) -> bool:
        if self._final:
            return True
        return self._t_draft is not None and self._clock() - self._t_draft >= self._hold_s

    def _gate_wait(self) -> float:
        if self._final or self._t_draft is None:
            return 1.0                          # changes wake us up; this is only a safety net
        return min(1.0, max(0.005, self._t_draft + self._hold_s - self._clock()))

    def _await_commit(self):
        """Wait until the reply can begin; returns its frame header, or None if there is no reply to send."""
        with self.cv:
            while True:
                if self.dead or self._failed or self.session.aborted:
                    return None
                self.session.poll()                 # a voice that has gone quiet is replaced, not waited for
                error = self.session.head_error()
                if error is not None:
                    self.voice_error = error
                    self._voice_dead = True
                    self.session.abort()
                    return None
                ready = self.session.peek()
                if ready is not None and (self._decided or self._gate_open()):
                    self._decided = True
                    self.voice, self.tts_ms = ready
                    first = self._t_draft if self._t_draft is not None else self._clock()
                    self.chat_ms = int((first - self._t0) * 1000)
                    return self._build_header(self.voice, " ".join(self._spoken + self._draft), self.chat_ms,
                                              self.tts_ms, self._get_actions())
                if self._finished and self.session.is_empty():
                    return None
                self.cv.wait(self._gate_wait())

    def _write(self) -> None:
        try:
            header = self._await_commit()
            if header is None:
                return
            for tag, data, first in self.session.pieces():
                if not self.started:
                    self._transport.begin()
                    self.started = True
                    self._transport.write(header + data)
                    self.first_audio_ms = self._since_start_ms()
                else:
                    self._transport.write(data)
                if first and isinstance(tag, int):
                    self._round_audio.setdefault(tag, self._since_start_ms())
                self.audio += data
            if self.started:
                if self.session.error is not None:
                    print(f"[tts] stopped mid-reply: {self.session.error}")
                    with self.cv:
                        self._stopped = True            # nobody can hear the rest: stop Gemini and the voice,
                    self.session.abort()                # and do it before the speaker is told the reply is over
                self._transport.end()
        except Exception as exc:               # the speaker hung up (broken pipe), or the connection dropped
            self.dead = True
            self.session.abort()
            print(f"[turn] the speaker hung up mid-reply ({type(exc).__name__})")
        finally:
            with self.cv:
                self.cv.notify_all()
