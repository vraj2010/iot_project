"""Talk to the server the way the S3 does, from any PC on the same Wi-Fi.

    python client.py --find                      is the server answering discovery?
    python client.py --say "what's my name?"     typed question  -> /say (voice reply saved to reply.mp3/.wav)
    python client.py --ask question.wav          recorded question -> /ask
    python client.py --chat "hi"                 text-only reply  -> /chat
    python client.py --say "play kesariya" --music          as a speaker that can play music
    python client.py --song 12                   download song 12 as the speaker gets it (song.mp3)
    python client.py --listen                    be the speaker for the laptop's mic: wait for the wake word,
                                                 then save each answer (reply.mp3). Ctrl+C to stop

Uses DEVICE_TOKEN from .env. Add --host 192.168.x.x to skip discovery.
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import time
import wave

import config
import frames


def find_server(timeout: float = 3.0) -> tuple[str, int] | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        # Windows: a probe that finds nothing listening would make the next recvfrom fail with WinError 10054
        s.ioctl(socket.SIO_UDP_CONNRESET, False)
    s.settimeout(0.5)
    end = time.time() + timeout
    targets = ["255.255.255.255", "127.0.0.1"]
    while time.time() < end:
        for t in targets:
            try:
                s.sendto(b"I2SONIC?", (t, config.UDP_PORT))
            except OSError:
                pass
        try:
            data, addr = s.recvfrom(64)
        except (socket.timeout, ConnectionResetError):
            continue                                       # nobody answered yet: ask again
        if data.startswith(b"I2SONIC!"):
            port = int(data[8:] or config.HTTP_PORT)
            return addr[0], port
    return None


def request(host: str, port: int, path: str, body: bytes | None, ctype: str = "",
            music_state: str | None = None) -> tuple[int, bytes, float]:
    conn = http.client.HTTPConnection(host, port, timeout=60)
    t0 = time.time()
    headers = {"X-Device-Token": config.DEVICE_TOKEN}
    if ctype:
        headers["Content-Type"] = ctype
    if music_state:
        headers["X-Music-State"] = music_state
    conn.request("POST" if body is not None else "GET", path, body=body, headers=headers)
    resp = conn.getresponse()
    first = resp.read1(65536) if hasattr(resp, "read1") else resp.read(1)
    first_at = time.time() - t0
    data = first + resp.read()
    conn.close()
    return resp.status, data, first_at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, default=config.HTTP_PORT)
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--say")
    ap.add_argument("--chat")
    ap.add_argument("--ask", metavar="WAV")
    ap.add_argument("--music", nargs="?", const="state=stopped", metavar="STATE",
                    help='send X-Music-State, e.g. "state=playing;track=12;pos=30;vol=70"')
    ap.add_argument("--song", type=int, metavar="ID", help="fetch /music/<id> into song.mp3")
    ap.add_argument("--listen", action="store_true", help="wait for the wake word on the laptop's mic, like the S3")
    ap.add_argument("--out", default="reply")
    args = ap.parse_args()

    host, port = args.host, args.port
    if not host:
        found = find_server()
        if not found:
            print("No server answered on UDP", config.UDP_PORT, "- is server.py running? Firewall?")
            sys.exit(1)
        host, port = found
        print(f"[found] server at {host}:{port}")
    if args.find:
        return

    if args.song is not None:
        status, data, t = request(host, port, f"/music/{args.song}", None)
        if status != 200:
            print(f"HTTP {status}: {data[:300].decode(errors='replace')}")
            sys.exit(1)
        open("song.mp3", "wb").write(data)
        print(f"[saved] song.mp3, {len(data) / 1024:.0f} KB, first bytes after {t * 1000:.0f} ms")
        return
    if args.listen:
        return listen(host, port, args.music, args.out)
    if args.chat:
        status, data, t = request(host, port, "/chat", json.dumps({"text": args.chat}).encode(), "application/json",
                                  args.music)
        print(status, json.loads(data))
        return
    if args.say:
        path, body, ctype = "/say", json.dumps({"text": args.say}).encode(), "application/json"
    elif args.ask:
        path, body, ctype = "/ask", open(args.ask, "rb").read(), "application/octet-stream"
    else:
        ap.print_help()
        return

    status, data, first_at = request(host, port, path, body, ctype, args.music)
    if status != 200:
        print(f"HTTP {status}: {data.decode(errors='replace')}")
        sys.exit(1)
    show(data, first_at, args.out)


def listen(host: str, port: int, music_state: str | None, out: str) -> None:
    """What the S3 does with LAPTOP_MIC 1: an idle GET /listen until the wake word, then /listen?wake=n."""
    print("[listening] say the wake word near the laptop (Ctrl+C to stop)")
    try:
        while True:
            status, data, _ = request(host, port, "/listen", None)
            if status == 204:
                continue                                   # nothing in LISTEN_HOLD_S: ask again
            if status != 202:
                print(f"HTTP {status}: {data.decode(errors='replace')}")
                sys.exit(1)
            wake = json.loads(data)["id"]
            print(f"[wake] {wake}: talk now")
            status, data, first_at = request(host, port, f"/listen?wake={wake}", None,
                                             music_state=music_state or "state=stopped;vol=70")
            if status == 202:
                print("[heard] nothing after the wake word")
            elif status == 200:
                show(data, first_at, out)
            else:
                print(f"HTTP {status}: {data.decode(errors='replace')}")
    except KeyboardInterrupt:
        print()


def show(data: bytes, first_at: float, out: str) -> None:
    f = frames.parse(data)
    if f["heard"] is not None:
        print(f"[heard] {f['heard'] or '(nothing clear)'}")
    print(f"nexa> {f['text']}")
    if f.get("action"):
        print(f"[action] {json.dumps(f['action'], ensure_ascii=False)}")
    print(f"[time] first bytes {first_at * 1000:.0f} ms | Whisper {f['stt_ms']} ms + Gemini {f['chat_ms']} ms "
          f"+ voice {f['tts_ms']} ms | {len(f['audio']) / 1024:.0f} KB of audio")
    if f["format"] == frames.FORMAT_MP3:
        out = out + ".mp3"
        open(out, "wb").write(f["audio"])
    else:
        out = out + ".wav"
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(f["rate"])
            w.writeframes(f["audio"])
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
