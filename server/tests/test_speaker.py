"""speaker.py (the laptop as the speaker), ptt.py (the board's BOOT button) and the wake word being optional."""

import http.server
import json
import struct
import threading
import time
import unittest
from unittest import mock

import numpy as np

import _path  # noqa: F401

import config  # noqa: E402
import frames  # noqa: E402
import ptt  # noqa: E402
import speaker  # noqa: E402
from ears.listener import Ears, NoWake  # noqa: E402
from ears.mic import FRAME  # noqa: E402


class Out:
    """The audio thread's outdata buffer."""

    def __init__(self, n):
        self.a = np.zeros((n, 2), np.int16)

    def __getitem__(self, key):
        return self.a[key]

    def __setitem__(self, key, value):
        self.a[key] = value


def pull(mixer, n=100):
    out = Out(n)
    mixer._callback(out, n, None, None)
    return out.a[:, 0].copy()


def tone(n, value=10000):
    return np.full(n, value, np.int16)


class MixerTest(unittest.TestCase):
    def mixer(self, volume=100):
        return speaker.Mixer(volume=volume)

    def test_voice_passes_through_at_full_volume(self):
        m = self.mixer()
        m.push_voice(tone(100, 1234))
        self.assertTrue((pull(m) == 1234).all())

    def test_volume_scales_everything(self):
        m = self.mixer(volume=50)
        m.push_voice(tone(100, 2000))
        self.assertTrue((pull(m) == 1000).all())

    def test_silence_when_nothing_is_queued(self):
        self.assertTrue((pull(self.mixer()) == 0).all())

    def test_music_plays_at_its_own_level(self):
        m = self.mixer()
        m.push_music(tone(100, 10000))
        out = pull(m)
        self.assertTrue((out == int(10000 * speaker.Mixer.MUSIC_GAIN)).all())
        self.assertEqual(m.music_played, 100)

    def test_music_ducks_while_she_answers_and_comes_back(self):
        m = self.mixer()
        m.push_music(tone(4000, 10000))
        m.replying = True
        for _ in range(5):
            pull(m, 200)                                   # the ramp down
        out = pull(m, 200)
        self.assertTrue((out == int(10000 * speaker.Mixer.MUSIC_GAIN * speaker.Mixer.DUCK)).all())
        m.replying = False
        for _ in range(5):
            pull(m, 200)
        self.assertTrue((pull(m, 200) == int(10000 * speaker.Mixer.MUSIC_GAIN)).all())

    def test_voice_and_ducked_music_add(self):
        m = self.mixer()
        m.push_music(tone(2000, 10000))
        m.replying = True
        for _ in range(3):
            pull(m, 200)
        m.push_voice(tone(200, 1000))
        out = pull(m, 200)
        self.assertEqual(int(out[-1]), 1000 + int(10000 * speaker.Mixer.MUSIC_GAIN * speaker.Mixer.DUCK))

    def test_music_stops_while_you_talk_and_loses_nothing(self):
        m = self.mixer()
        m.push_music(tone(2000, 10000))
        m.listening = True
        pull(m, 200)                                       # the ramp down eats one block
        queued = m._music_n
        for _ in range(3):
            self.assertTrue((pull(m, 200) == 0).all())
        self.assertEqual(m._music_n, queued)               # not consumed while muted
        m.listening = False
        for _ in range(3):
            pull(m, 200)
        self.assertGreater(pull(m, 200)[-1], 0)

    def test_paused_music_is_not_consumed(self):
        m = self.mixer()
        m.push_music(tone(2000, 10000))
        m.music_paused = True
        pull(m, 200)
        queued = m._music_n
        pull(m, 200)
        self.assertEqual(m._music_n, queued)

    def test_clear_voice_drops_the_rest_of_the_sentence(self):
        m = self.mixer()
        m.push_voice(tone(1000, 5000))
        pull(m, 100)
        m.clear_voice()
        self.assertTrue((pull(m, 100) == 0).all())

    def test_output_is_clipped_not_wrapped(self):
        m = self.mixer()
        m.push_voice(tone(100, 30000))
        m.push_music(tone(100, 30000))
        self.assertTrue((pull(m) > 0).all())

    def test_chime_is_short_and_not_loud(self):
        c = speaker.chime()
        self.assertLess(c.size / speaker.OUT_RATE, 0.3)
        self.assertLess(int(np.abs(c).max()), 12000)


class Resp:
    """http.client's response, handing out the bytes in small pieces like a slow network."""

    def __init__(self, data, step=3):
        self.data, self.step = data, step

    def read1(self, n):
        piece, self.data = self.data[:min(n, self.step)], self.data[min(n, self.step):]
        return piece


class HeaderTest(unittest.TestCase):
    def check(self, header, audio, expect):
        bs = speaker.ByteStream(Resp(header + audio))
        f = speaker.read_reply_header(bs)
        for k, v in expect.items():
            self.assertEqual(f[k], v)
        self.assertEqual(b"".join(bs.chunks()), audio)

    def test_i2s4_with_music_action(self):
        h = frames.reply_header("play kesariya", "Playing Kesariya.", 1, 10, 20, 30, "mp3",
                                {"music": "play", "track": 12})
        self.check(h, b"\x01\x02" * 50, {"heard": "play kesariya", "text": "Playing Kesariya.",
                                         "action": {"music": "play", "track": 12}, "format": frames.FORMAT_MP3})

    def test_i2s3_and_i2s2(self):
        self.check(frames.ask_header("hi", "Hello there.", 1, 1, 1, 1, "pcm"), b"abcd" * 30,
                   {"heard": "hi", "text": "Hello there.", "rate": 16000})
        self.check(frames.say_header("Hello.", 1, 1, 1, "pcm"), b"xyxy" * 30, {"text": "Hello.", "heard": None})

    def test_unicode_text(self):
        h = frames.reply_header("ગુજરાતી", "નમસ્તે", 1, 1, 1, 1, "pcm", None)
        self.check(h, b"zz", {"heard": "ગુજરાતી", "text": "નમસ્તે"})

    def test_not_a_frame_and_truncated(self):
        with self.assertRaises(ValueError):
            speaker.read_reply_header(speaker.ByteStream(Resp(b"HTTP/1.1 200 OK")))
        with self.assertRaises(ValueError):
            speaker.read_reply_header(speaker.ByteStream(Resp(b"I2S4\x00\x00")))


class PttTest(unittest.TestCase):
    def test_parse_line(self):
        self.assertEqual(ptt.parse_line("PTT 1"), "down")
        self.assertEqual(ptt.parse_line("PTT 0\r\n"), "up")
        self.assertEqual(ptt.parse_line("\x00\xffPTT 1"), "down")      # a stray byte after a reset
        for junk in ("", "Play", "Device connected", "PTT", "PTT 2", "PTT x", "friday PTT?"):
            self.assertIsNone(ptt.parse_line(junk), junk)

    def test_named_port_is_used_as_given(self):
        self.assertEqual(ptt.find_port("COM7"), "COM7")


class NoWakeTest(unittest.TestCase):
    def test_button_only_ears(self):
        class Mic:
            name = "fake"

            def start(self): pass

            def read(self, timeout=1.0):
                time.sleep(0.01)

            def close(self): pass

        class Vad:
            def reset(self): pass

            def prob(self, frame): return 0.0

        with mock.patch.object(config, "WAKE_ENABLED", False), mock.patch("builtins.print"):
            ears = Ears(mic=Mic(), vad=Vad())
            ears.start()
            self.addCleanup(ears.stop)
            self.assertIsInstance(ears.wake, NoWake)
            self.assertEqual(ears.phrase, "")
            for _ in range(10):
                ears.feed(np.ones(FRAME, np.int16) * 20000)             # loud: still never wakes
            self.assertEqual(ears.state, "idle")
            self.assertEqual(ears.wait_for_wake(0), None)
            self.assertTrue(ears.wake_now())                            # the button does
            self.assertEqual(ears.wait_for_wake(0), 1)


# ---- the whole speaker against a fake server


class FakeFriday(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.log = []
        self.wake_answers = [202]                  # what POST /listen/wake says, in turn
        self.reply = b""
        self.music = b"\x05" * 30
        self.next_music = None                     # (track id, bytes) for /music/next, or None = 204
        self.idle_wakes = 1


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"                  # the body ends when the connection closes

    def log_message(self, *a):
        pass

    def reply(self, code, body=b"", **headers):
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k.replace("_", "-"), v)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        srv = self.server
        srv.log.append(("GET", self.path, self.headers.get("X-Device-Token"), self.headers.get("X-Music-State")))
        if self.path == "/listen":
            if srv.idle_wakes:
                srv.idle_wakes -= 1
                return self.reply(202, json.dumps({"event": "wake", "id": 1}).encode())
            time.sleep(0.05)
            return self.reply(204)
        if self.path.startswith("/listen?wake="):
            self.send_response(200)
            self.end_headers()
            data = srv.reply
            for i in range(0, len(data), 7):               # a few bytes at a time, with gaps
                self.wfile.write(data[i:i + 7])
                self.wfile.flush()
                time.sleep(0.002)
            return
        if self.path.startswith("/music/next"):
            if not srv.next_music:
                return self.reply(204)
            tid, body = srv.next_music
            srv.next_music = None
            return self.reply(200, body, X_Track_Id=str(tid), X_Track_Title="Next%20Song")
        if self.path.startswith("/music/"):
            tid = self.path.split("/")[2]
            return self.reply(200, srv.music, X_Track_Id=tid, X_Track_Title="Kesariya%20by%20Arijit")
        self.reply(404)

    def do_POST(self):
        srv = self.server
        srv.log.append(("POST", self.path, self.headers.get("X-Device-Token"), None))
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        code = srv.wake_answers.pop(0) if len(srv.wake_answers) > 1 else srv.wake_answers[0]
        self.reply(code, json.dumps({"ok": code == 202}).encode())


class FakeMp3:
    """Stands in for PyAV: every byte becomes one sample worth byte*100."""

    def __init__(self, rate=None):
        pass

    def feed(self, data):
        return [np.frombuffer(data, np.uint8).astype(np.int16) * 100]

    def flush(self):
        return []


class FakePcm:
    def __init__(self, src_rate=16000, rate=None):
        self.src, self.left = src_rate, b""

    def feed(self, data):
        data = self.left + data
        n = len(data) // 2 * 2
        self.left = data[n:]
        return [np.frombuffer(data[:n], "<i2").copy()] if n else []

    def flush(self):
        return []


class TestMixer(speaker.Mixer):
    """No sound card: a thread plays the part of the audio callback and keeps what came out."""

    def ensure_open(self):
        return True

    latency = 0.0

    def start_drain(self):
        self.heard = []
        self._stop = False

        def run():
            while not self._stop:
                self.heard.append(pull(self, 64))
                time.sleep(0.001)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop_drain(self):
        self._stop = True
        self._t.join()

    def voice_out(self):
        return np.concatenate(self.heard)


class SpeakerTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(config, "DEVICE_TOKEN", "tok")
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch("builtins.print")
        p.start()
        self.addCleanup(p.stop)
        self.srv = FakeFriday()
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.server = ["127.0.0.1", self.srv.server_address[1]]
        self.mixer = TestMixer(volume=100)
        self.mixer.start_drain()
        self.addCleanup(self.mixer.stop_drain)
        self.music = speaker.MusicPlayer(self.mixer, self.server, decoder=FakeMp3, log=lambda *a: None)
        self.speaker = speaker.Speaker(self.server, self.mixer, self.music, mp3_decoder=FakeMp3,
                                       pcm_decoder=FakePcm, log=lambda *a: None)

    def pcm_reply(self, samples, action=None, text="Hello Deep."):
        head = frames.reply_header("what time is it", text, 1, 1, 1, 1, "pcm", action)
        return head + samples.astype("<i2").tobytes()

    def test_a_button_press_becomes_a_spoken_answer(self):
        voice = (np.arange(1, 801) % 300 + 50).astype(np.int16)
        self.srv.reply = self.pcm_reply(voice)
        self.speaker.listen_once()
        self.assertEqual(self.speaker.phase, "replying")
        time.sleep(0.1)
        out = self.mixer.voice_out()
        out = out[out != 0]
        chime = speaker.chime().size
        self.assertTrue(np.array_equal(out[-voice.size:], voice), "her voice came out whole and in order")
        self.assertGreaterEqual(out.size, voice.size + chime // 2)       # the chime came first
        self.assertFalse(self.mixer.listening)
        self.assertFalse(self.mixer.replying)
        gets = [e for e in self.srv.log if e[0] == "GET"]
        self.assertEqual(gets[0][1], "/listen")
        self.assertEqual(gets[1][1], "/listen?wake=1")
        self.assertTrue(all(e[2] == "tok" for e in gets), "every request carries the device token")
        self.assertIn("state=stopped", gets[1][3])
        self.assertIn("vol=100", gets[1][3])

    def test_the_next_poll_after_an_answer_is_idle(self):
        self.srv.reply = self.pcm_reply(tone(100))
        self.speaker.listen_once()
        self.speaker.listen_once()                      # 204: nothing happened
        self.assertEqual(self.speaker.phase, "idle")

    def test_actions_set_volume_and_start_music(self):
        self.srv.music = b"\x05" * 150000                  # a long song: still playing when we look
        self.srv.reply = self.pcm_reply(tone(100, 100), {"music": "play", "track": 5, "volume": 40})
        self.speaker.listen_once()
        self.assertEqual(self.mixer.volume, 40)
        deadline = time.time() + 3
        while self.music.track != 5 or self.mixer.music_played == 0:
            self.assertLess(time.time(), deadline, "the song never started")
            time.sleep(0.02)
        self.assertEqual(self.music.state, "playing")
        self.assertIn(("GET", "/music/5", "tok", None), self.srv.log)
        header = self.music.header()
        self.assertTrue(header.startswith("state=playing;track=5;pos="))
        self.assertTrue(header.endswith("vol=40"))
        self.music.pause()
        self.assertEqual(self.music.state, "paused")
        self.assertIn("state=paused", self.music.header())
        self.music.resume()
        self.assertEqual(self.music.state, "playing")
        self.music.stop()
        self.assertEqual(self.music.header(), "state=stopped;vol=40")

    def test_the_next_song_starts_when_one_ends(self):
        self.srv.next_music = (9, b"\x07" * 60000)         # long enough to still be playing when we look
        self.music.play(5)
        deadline = time.time() + 5
        while self.music.track != 9:
            self.assertLess(time.time(), deadline, "never moved on to the next song")
            time.sleep(0.02)
        self.assertIn(("GET", "/music/next?after=5", "tok", None), self.srv.log)
        deadline = time.time() + 10
        while self.music.state != "stopped":              # then /music/next says 204: the queue is finished
            self.assertLess(time.time(), deadline, "never stopped after the last song")
            time.sleep(0.02)

    def test_cancelled_by_the_button_the_rest_of_the_reply_is_dropped(self):
        self.srv.reply = self.pcm_reply(tone(40000, 700))
        self.speaker.cancel.set()                         # the press arrives while she is talking
        orig = self.speaker.cancel.clear
        with mock.patch.object(self.speaker.cancel, "clear", lambda: None):   # keep the press, as if it came mid-reply
            self.speaker.listen_once()
        orig()
        time.sleep(0.1)
        voice = self.mixer.voice_out()
        self.assertLess(int((voice == 700).sum()), 40000)

    def test_press_retries_while_the_server_is_still_busy(self):
        self.srv.wake_answers = [409, 409, 202]
        self.speaker.press()
        posts = [e for e in self.srv.log if e[0] == "POST"]
        self.assertEqual([e[1] for e in posts], ["/listen/wake"] * 3)

    def test_press_while_already_listening_does_nothing(self):
        self.speaker.phase = "listening"
        self.speaker.press()
        self.assertEqual([e for e in self.srv.log if e[0] == "POST"], [])

    def test_press_while_she_talks_cuts_her_off_first(self):
        self.speaker.phase = "replying"
        self.mixer.push_voice(tone(100000, 900))
        self.speaker.press()
        self.assertTrue(self.speaker.cancel.is_set())
        time.sleep(0.05)
        self.assertEqual(self.mixer._voice_n, 0)

    def test_a_dead_sound_output_means_text_only(self):
        self.srv.reply = self.pcm_reply(tone(100))
        with mock.patch.object(TestMixer, "ensure_open", lambda self: False):
            self.speaker.listen_once()                    # must not hang or raise
        time.sleep(0.4)                                   # the chime plays out
        self.assertEqual(self.mixer._voice_n, 0, "her voice wasn't queued")


if __name__ == "__main__":
    unittest.main()
