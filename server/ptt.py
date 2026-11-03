"""The board's BOOT button, read over its USB cable.

friday_speaker.ino prints "PTT 1" to the serial port when BOOT goes down and "PTT 0" when it comes up.
speaker.py reads those lines here and tells the server to listen. Close the Arduino Serial Monitor first:
only one program can hold a COM port.

    python ptt.py             find the board's port and print every press (Ctrl+C to stop)
    python ptt.py COM5        the same, on a port you name
"""

from __future__ import annotations

import sys
import threading
import time

# USB-serial chips on ESP32 boards: Silicon Labs CP210x, WCH CH340/CH910, FTDI, Espressif's own USB.
_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}
_WORDS = ("cp210", "ch340", "ch910", "usb-serial", "usb serial", "silicon labs", "esp32")


def parse_line(line: str) -> str | None:
    """"PTT 1" -> "down", "PTT 0" -> "up", anything else (boot messages, logs) -> None."""
    line = line.strip()
    at = line.rfind("PTT ")                      # tolerate a stray byte or two in front
    if at < 0:
        return None
    tail = line[at + 4:].strip()
    return {"1": "down", "0": "up"}.get(tail)


def _serial():
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial isn't installed: pip install pyserial") from exc
    return serial, list_ports


def list_candidates() -> list[str]:
    """COM ports that look like an ESP32 board's USB-serial chip."""
    _, list_ports = _serial()
    out = []
    for p in list_ports.comports():
        text = f"{p.description or ''} {p.manufacturer or ''}".lower()
        if p.vid in _VIDS or any(w in text for w in _WORDS):
            out.append(p.device)
    return out


def find_port(spec: str = "auto") -> str | None:
    """`spec` is a port name (COM5), or "auto" for the first port that looks like an ESP32 board."""
    spec = (spec or "auto").strip()
    if spec.lower() != "auto":
        return spec
    ports = list_candidates()
    return ports[0] if ports else None


class SerialButton:
    """Reads the board's serial port in a background thread and calls on_press / on_release. It reconnects by
    itself when the cable is pulled out and put back."""

    def __init__(self, port: str = "auto", on_press=None, on_release=None, baud: int = 115200, log=print) -> None:
        self.port, self.baud, self.log = port, baud, log
        self.on_press, self.on_release = on_press, on_release
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        _serial()                                # fail here, not in the thread, if pyserial is missing
        self._thread = threading.Thread(target=self._run, name="ptt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _fire(self, cb) -> None:
        if cb:
            threading.Thread(target=cb, daemon=True).start()     # never hold up the serial reader

    def _run(self) -> None:
        serial, _ = _serial()
        warned = ""
        while not self._stop.is_set():
            port = find_port(self.port)
            if not port:
                if warned != "none":
                    self.log("[ptt] no ESP32 serial port found (is the USB cable plugged in? the Arduino Serial "
                             "Monitor closed?). Trying again every few seconds; Enter works meanwhile")
                    warned = "none"
                time.sleep(3)
                continue
            try:
                ser = serial.Serial()
                ser.port, ser.baudrate, ser.timeout = port, self.baud, 1
                ser.dtr = False                  # opening a port normally resets an ESP32 board: don't
                ser.rts = False
                ser.open()
            except (serial.SerialException, OSError) as exc:
                if warned != port:
                    self.log(f"[ptt] can't open {port}: {exc} (another program using it?)")
                    warned = port
                time.sleep(3)
                continue
            warned = ""
            self.log(f"[ptt] reading the BOOT button on {port}")
            try:
                while not self._stop.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    event = parse_line(raw.decode("utf-8", "replace"))
                    if event == "down":
                        self._fire(self.on_press)
                    elif event == "up":
                        self._fire(self.on_release)
            except (serial.SerialException, OSError):
                self.log(f"[ptt] lost {port}, looking again")
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
            time.sleep(1)


def main() -> None:
    spec = sys.argv[1] if len(sys.argv) > 1 else "auto"
    try:
        print("ports that look like an ESP32 board:", list_candidates() or "none")
    except RuntimeError as exc:
        print(exc)
        return
    btn = SerialButton(spec, on_press=lambda: print("BOOT down"), on_release=lambda: print("BOOT up"))
    btn.start()
    print("press the BOOT button on the board (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
