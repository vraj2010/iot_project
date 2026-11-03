// ============================================================
//  Friday firmware: private settings
//  Copy this file to secrets.h and fill it in. git ignores secrets.h.
// ============================================================
#pragma once

// The Wi-Fi the S3 joins (2.4 GHz; a phone hotspot works).
#define WIFI_SSID        "your-wifi-name"
#define WIFI_PASSWORD    "your-wifi-password"

// Only if the S3 can't find the laptop by itself (UDP discovery): the laptop's LAN IP,
// e.g. "192.168.1.20". Leave "" to rely on discovery.
#define LAPTOP_IP_FIXED  ""

// The backup brain: your Cloudflare Worker (see ../../worker/README.md).
#define WORKER_URL       "https://i2sonic.your-subdomain.workers.dev"

// Same value in the Worker secret AND in server/.env (DEVICE_TOKEN).
#define DEVICE_TOKEN     "PASTE-YOUR-DEVICE-TOKEN-HERE"
