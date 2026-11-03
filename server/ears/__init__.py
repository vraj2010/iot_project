"""Friday's ears on the laptop: its own microphone instead of the speaker's.

    laptop mic ──80 ms frames──▶ wake word ──"hey jarvis"──▶ record ──silence──▶ the question
       (mic.py)                   (wake.py)                   (vad.py + endpoint.py)

listener.Ears runs that loop on a thread and hands each question to the speaker, which waits for it with
GET /listen (server.py). Nothing is recorded or sent anywhere until the wake word is heard.
"""

from .listener import Ears

__all__ = ["Ears"]
