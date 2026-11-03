/*
  Nexa speaker - ESP32 (WROOM-32) as a general Bluetooth speaker.

  Any phone, tablet or laptop can pair with "Nexa" and play audio through
  the MAX98357A amp. Your laptop pipeline (TTS and music) works the same way,
  as just one more source.

  NOTE: Bluetooth Classic A2DP on this chip connects to ONE device at a time.
  Use the button to switch between devices (see below).

  Libraries (both from Library Manager or GitHub, by pschatzmann):
    - ESP32-A2DP
    - arduino-audio-tools
  Board: "ESP32 Dev Module" (ESP32 Arduino core 2.x or 3.x)

  Wiring (verify on your board; avoid GPIO 0, 2, 12, 15 and 6-11):
    MAX98357A BCLK -> GPIO 26
    MAX98357A LRC  -> GPIO 25
    MAX98357A DIN  -> GPIO 22
    MAX98357A SD   -> GPIO 27   (driven HIGH here; floating = amp shut down)
    MAX98357A GAIN -> GND       (12 dB, mono)
    MAX98357A VIN  -> 5V (or 3V3 if 5V sounds hissy), GND -> GND
    Speaker: 4 ohm (two 8 ohm in parallel) on SPK+ / SPK-, never to GND
    Button: GPIO 13 -> GND (internal pull-up, no resistor needed)

  Button:
    short press  = play / pause the connected device
    long press   = disconnect the current device and open pairing, so the
                   next device (phone, laptop...) can connect

  Onboard LED (GPIO 2): on = a device is connected, blinking = waiting.

  Push-to-talk: BOOT (GPIO 0) prints "PTT 1" when pressed and "PTT 0" when released on the USB serial
  port (115200). speaker.py on the laptop reads it and starts listening. Close the Arduino Serial Monitor
  while speaker.py runs (only one program can use the port). Don't hold BOOT while pressing EN/reset.
*/

#include "AudioTools.h"
#include "BluetoothA2DPSink.h"

#define PIN_BCLK   26
#define PIN_LRC    25
#define PIN_DIN    22
#define PIN_AMP_SD 27
#define PIN_BUTTON 13
#define PIN_LED    2     // onboard LED on most ESP32 DevKits
#define PIN_PTT    0     // BOOT button: push-to-talk, sent to the laptop over USB serial

#define DEVICE_NAME      "Nexa"
#define START_VOLUME     80      // 0-127; connected devices then control it
#define LONG_PRESS_MS    1500
#define RECONNECT_PAUSE  30000   // after a switch, don't auto-grab the old device

I2SStream i2s;
BluetoothA2DPSink a2dp_sink(i2s);

bool wasConnected = false;
unsigned long pressStart = 0;
bool pressing = false;
bool longHandled = false;
unsigned long reconnectResumeAt = 0;
bool pttDown = false;

void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_PTT, INPUT_PULLUP);

  // Wake the amp (SD floating = shutdown)
  pinMode(PIN_AMP_SD, OUTPUT);
  digitalWrite(PIN_AMP_SD, HIGH);

  // I2S output to the MAX98357A
  auto cfg = i2s.defaultConfig();
  cfg.pin_bck  = PIN_BCLK;
  cfg.pin_ws   = PIN_LRC;
  cfg.pin_data = PIN_DIN;
  i2s.begin(cfg);

  // Reconnects to the last device after power-up; any other device can still pair.
  a2dp_sink.set_auto_reconnect(true);
  a2dp_sink.start(DEVICE_NAME);
  a2dp_sink.set_volume(START_VOLUME);

  Serial.println("Nexa speaker ready. Pair 'Nexa' from any device's Bluetooth settings.");
}

// BOOT = push-to-talk. The laptop (speaker.py) reads these two lines from the USB serial port.
void handlePTT() {
  static unsigned long changedAt = 0;
  bool down = (digitalRead(PIN_PTT) == LOW);
  if (down != pttDown && millis() - changedAt > 30) {   // 30 ms debounce
    pttDown = down;
    changedAt = millis();
    Serial.println(down ? "PTT 1" : "PTT 0");
  }
}

void handleButton() {
  bool down = (digitalRead(PIN_BUTTON) == LOW);

  if (down && !pressing) {
    pressing = true;
    longHandled = false;
    pressStart = millis();
  }

  if (down && pressing && !longHandled && millis() - pressStart >= LONG_PRESS_MS) {
    longHandled = true;
    Serial.println("Long press: disconnecting, open for the next device");
    a2dp_sink.set_auto_reconnect(false);   // stop it grabbing the old device back
    reconnectResumeAt = millis() + RECONNECT_PAUSE;
    if (a2dp_sink.is_connected()) a2dp_sink.disconnect();
  }

  if (!down && pressing) {
    pressing = false;
    if (!longHandled && millis() - pressStart > 40) {   // short press, debounced
      if (a2dp_sink.get_audio_state() == ESP_A2D_AUDIO_STATE_STARTED) {
        a2dp_sink.pause();
        Serial.println("Pause");
      } else {
        a2dp_sink.play();
        Serial.println("Play");
      }
    }
  }
}

void loop() {
  handlePTT();
  handleButton();

  // Re-enable auto-reconnect a while after a manual switch
  if (reconnectResumeAt && millis() > reconnectResumeAt) {
    a2dp_sink.set_auto_reconnect(true);
    reconnectResumeAt = 0;
  }

  bool connected = a2dp_sink.is_connected();
  if (connected != wasConnected) {
    wasConnected = connected;
    Serial.println(connected ? "Device connected" : "Device disconnected");
  }

  // LED: solid when connected, blink when waiting
  if (connected) digitalWrite(PIN_LED, HIGH);
  else digitalWrite(PIN_LED, (millis() / 500) % 2);

  delay(10);
}
