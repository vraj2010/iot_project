// Wi-Fi, and finding the laptop that runs friday-local.
#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "esp_err.h"

// Connects to Wi-Fi and blocks until an IP arrives (or the timeout passes).
esp_err_t net_wifi_start(void);
bool      net_is_online(void);

// Broadcast "I2SONIC?" and wait up to timeout_ms for friday-local to answer
// "I2SONIC!<port>". On success fills in the laptop's IP and HTTP port.
esp_err_t net_find_laptop(int timeout_ms, char *ip, size_t ip_len, int *port);

// Quick TCP check that something is listening at ip:port. On the home Wi-Fi
// this takes a few milliseconds; it gives up after timeout_ms.
bool net_can_connect(const char *ip, int port, int timeout_ms);
