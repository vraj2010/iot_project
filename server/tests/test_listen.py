"""GET /listen and POST /listen/wake (the laptop as Friday's ears), and commands answered without Gemini."""

import contextlib
import datetime
import http.client
import json
import threading
import time
import unittest
from unittest import mock

import numpy as np

import _path  # noqa: F401

import config  # noqa: E402
import frames  # noqa: E402
import gemini  # noqa: E402
import intents  # noqa: E402
import server  # noqa: E402
from brainfakes import script  # noqa: E402
from ears.listener import Ears  # noqa: E402
from ears.mic import FRAME  # noqa: E402
from test_intents import library  # noqa: E402
from test_server_stream import END, Base  # noqa: E402
from voicefakes import wait_until  # noqa: E402


class FakeEars:
    source = "wake"

    def __init__(self, wake=None, question=None, wake_ok=True):
        self.wake, self.question, self.wake_ok = wake, question, wake_ok
        self.turns, self.asked, self.pressed = 0, [], 0

    def wait_for_wake(self, timeout):
        return self.wake

    def wait_for_question(self, wake_id, timeout):
        self.asked.append(wake_id)
        return self.question

    def wake_now(self):
        self.pressed += 1
        return self.wake_ok

    @contextlib.contextmanager
    def busy(self):
        self.turns += 1
        yield

    def status(self):
        return {"state": "idle"}


def no_gemini(*a, **k):
    raise AssertionError("Gemini was asked")


class ListenBase(Base):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(config, "LISTEN_HOLD_S", 1)
        p.start()
        self.addCleanup(p.stop)
        self.use(None)

    def use(self, ears):
        old = server.Handler.ears
        server.Handler.ears = ears
        self.addCleanup(setattr, server.Handler, "ears", old)
        return ears

    def get(self, path, token="tok", headers=None, timeout=10):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        conn.request("GET", path, headers={**({"X-Device-Token": token} if token else {}), **(headers or {})})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data


class Listen(ListenBase):
    def test_needs_the_device_token(self):
        self.use(FakeEars())
        self.assertEqual(self.get("/listen", token=None)[0], 401)
        self.assertEqual(self.get("/listen", token="nope")[0], 401)

    def test_no_ears_says_so(self):
        status, body = self.get("/listen")
        self.assertEqual(status, 503)
        self.assertIn("EARS=0", json.loads(body)["error"])

    def test_idle_poll_nothing_heard_is_204(self):
        self.use(FakeEars(wake=None))
        self.assertEqual(self.get("/listen"), (204, b""))

    def test_idle_poll_wake_word_is_202_with_its_id(self):
        self.use(FakeEars(wake=3))
        status, body = self.get("/listen")
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(body), {"event": "wake", "id": 3, "source": "wake"})

    def test_awake_poll_answers_like_ask(self):
        ears = self.use(FakeEars(question=np.zeros(16000, np.int16)))
        fake, _ = script([("text", "It is sunny today."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            status, body = self.get("/listen?wake=3", headers={"X-Music-State": "state=stopped;vol=70"})
        self.assertEqual(status, 200)
        f = frames.parse(body)
        self.assertEqual((f["heard"], f["text"], f["stt_ms"]), ("what is the weather", "It is sunny today.", 111))
        self.assertEqual(f["audio"], self.audio_of("It is sunny today."))
        self.assertEqual(ears.asked, [3])
        self.assertEqual(ears.turns, 1)                             # deaf while she answers
        self.assertTrue(wait_until(lambda: self.assistant.logged))

    def test_awake_poll_nothing_said_is_a_cancel(self):
        self.use(FakeEars(question=None))
        status, body = self.get("/listen?wake=3")
        self.assertEqual((status, json.loads(body)), (202, {"event": "cancel"}))
        self.assertEqual(self.get("/listen?wake=x")[0], 400)

    def test_boot_button_asks_the_ears_to_listen(self):
        ears = self.use(FakeEars())
        self.assertEqual(self.post("/listen/wake", b"")[0], 202)
        ears.wake_ok = False
        self.assertEqual(self.post("/listen/wake", b"")[0], 409)
        self.assertEqual(ears.pressed, 2)
        self.use(None)
        self.assertEqual(self.post("/listen/wake", b"")[0], 503)

    def test_every_turn_mutes_the_ears(self):
        ears = self.use(FakeEars())
        fake, _ = script([("text", "Hi."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            self.assertEqual(self.say("hello")[0], 200)
        self.assertEqual(ears.turns, 1)

    def test_health_shows_the_ears(self):
        self.assertEqual(json.loads(self.get("/health")[1])["ears"], "off")


class LocalCommands(ListenBase):
    def setUp(self):
        super().setUp()
        self.assistant.router = intents.Router()
        self.assistant.jukebox = library()
        p = mock.patch.object(gemini, "chat_stream", no_gemini)
        p.start()
        self.addCleanup(p.stop)

    def test_a_spoken_command_is_answered_without_gemini(self):
        self.assistant.heard = "Pause the music."
        status, body = self.post("/ask", _wav(), {"X-Music-State": "state=playing;track=1;pos=30;vol=70"})
        self.assertEqual(status, 200)
        f = frames.parse(body)
        self.assertEqual((f["heard"], f["text"], f["action"]), ("Pause the music.", "Paused.", {"music": "pause"}))
        self.assertEqual(f["audio"], self.voice.audio("Paused."))
        self.assertEqual(self.assistant.legacy_calls, [])
        time.sleep(0.05)
        self.assertEqual(self.assistant.logged, [])                 # "pause" isn't worth remembering

    def test_a_typed_command_too(self):
        status, body = self.say("volume 40", {"X-Music-State": "state=playing;track=1;pos=30;vol=70"})
        f = frames.parse(body)
        self.assertEqual((f["text"], f["action"]), ("Volume 40 percent.", {"volume": 40}))

    def test_the_time_needs_no_music(self):
        with mock.patch.object(self.assistant.router, "_now", return_value=datetime.datetime(2026, 10, 24, 9, 30)):
            status, body = self.say("what time is it")
        self.assertEqual(frames.parse(body)["text"], "It's 9:30 AM.")

    def test_chat_answers_json(self):
        status, body = self.post("/chat", {"text": "next song"}, {"X-Music-State": "state=playing;track=1;pos=30;vol=70"})
        out = json.loads(body)
        self.assertTrue(out["reply"].startswith("Here's "))
        self.assertEqual(out["action"]["music"], "play")
        self.assertEqual(out["tools"], ["local:next"])

    def test_anything_else_still_goes_to_gemini(self):
        fake, calls = script([("text", "Why did the chicken..."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            status, body = self.say("tell me a joke")
        self.assertEqual(frames.parse(body)["text"], "Why did the chicken...")
        self.assertEqual(len(calls), 1)

    def test_a_broken_router_falls_back_to_gemini(self):
        self.assistant.router.route = mock.Mock(side_effect=RuntimeError("bug"))
        fake, calls = script([("text", "Paused it."), END])
        with mock.patch.object(gemini, "chat_stream", fake), mock.patch("builtins.print"):
            status, body = self.say("pause")
        self.assertEqual((status, len(calls)), (200, 1))


class WholeLoop(ListenBase):
    """The real Ears (with scripted models) behind the real HTTP server: wake, record, answer."""

    def test_wake_word_to_spoken_answer(self):
        class Script:
            value, phrase = 0.0, "hey jarvis"

            def score(self, frame):
                return self.value

            prob = score

            def reset(self):
                pass

        wake, vad = Script(), Script()
        ears = self.use(Ears(wake=wake, vad=vad))

        def feed(n, w=0.0, s=0.0):
            wake.value, vad.value = w, s
            for _ in range(n):
                ears.feed(np.zeros(FRAME, np.int16))
            wake.value = vad.value = 0.0

        idle = []
        t = threading.Thread(target=lambda: idle.append(self.get("/listen")))
        t.start()
        self.assertTrue(wait_until(lambda: ears.polling == 1))
        with mock.patch("builtins.print"):
            feed(1, w=0.9)                                         # "hey jarvis"
            t.join(5)
            status, body = idle[0]
            self.assertEqual(status, 202)
            wake_id = json.loads(body)["id"]

            answer = []
            fake, _ = script([("text", "It is sunny today."), END])
            with mock.patch.object(gemini, "chat_stream", fake):
                t = threading.Thread(target=lambda: answer.append(self.get(f"/listen?wake={wake_id}")))
                t.start()
                time.sleep(0.1)
                feed(5, s=0.9)                                     # "what's the weather"
                feed(int(config.EARS_SILENCE_MS / 80) + 1)         # and silence
                t.join(5)
        status, body = answer[0]
        self.assertEqual(status, 200)
        self.assertEqual(frames.parse(body)["text"], "It is sunny today.")
        self.assertTrue(ears.muted)                                # deaf until the speaker polls again


def _wav():
    from test_server_stream import wav

    return wav()


if __name__ == "__main__":
    unittest.main()
