#include "textin.h"

#include <stdio.h>
#include <string.h>

#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "sdkconfig.h"

#define CONSOLE_UART CONFIG_ESP_CONSOLE_UART_NUM
#define TEXTIN_LINE_MAX     480

static char   s_line[TEXTIN_LINE_MAX];
static size_t s_len;

void textin_init(void)
{
    // A UART driver gives blocking reads with a timeout. Route printf through it
    // too, so typed input and log output share the port cleanly.
    ESP_ERROR_CHECK(uart_driver_install(CONSOLE_UART, 512, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(CONSOLE_UART);
}

bool textin_poll(char *out, size_t len, int timeout_ms)
{
    TickType_t wait = pdMS_TO_TICKS(timeout_ms);

    for (;;) {
        uint8_t c;
        if (uart_read_bytes(CONSOLE_UART, &c, 1, wait) != 1) {
            return false;                   // nothing (more) typed for now
        }
        wait = 0;                           // drain what's already buffered, then return

        if (c == '\r' || c == '\n') {
            if (s_len == 0) {
                continue;                   // stray newline (monitors send CR+LF)
            }
            s_line[s_len] = '\0';
            strncpy(out, s_line, len - 1);
            out[len - 1] = '\0';
            s_len = 0;
            printf("\n");
            fflush(stdout);
            return true;
        }

        if (c == 0x08 || c == 0x7f) {       // Backspace / Delete
            if (s_len > 0) {
                s_len--;
                printf("\b \b");
                fflush(stdout);
            }
            continue;
        }

        if (c < 0x20) {
            continue;                       // ignore other control keys
        }

        if (s_len + 1 < TEXTIN_LINE_MAX && s_len + 1 < len) {
            s_line[s_len++] = (char)c;
            putchar(c);                     // the monitor doesn't echo what you type
            fflush(stdout);
        }
    }
}
