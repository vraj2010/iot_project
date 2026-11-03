// Typed input from the serial monitor (idf.py monitor), one line at a time.
#pragma once

#include <stdbool.h>
#include <stddef.h>

// Take over the console UART so keystrokes can be read. Call once at boot.
void textin_init(void);

// Collect keystrokes for up to `timeout_ms`, echoing them and handling Backspace.
// Returns true (and copies the line into `out`) when Enter completes a non-empty
// line; otherwise returns false and keeps the partial line for the next call.
// Polling like this lets one task watch both the keyboard and push-to-talk.
bool textin_poll(char *out, size_t len, int timeout_ms);
