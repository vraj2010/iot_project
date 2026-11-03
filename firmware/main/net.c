#include "net.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

#include "config.h"

static const char *TAG = "net";

#define WIFI_CONNECTED_BIT BIT0

static EventGroupHandle_t s_events;
static bool               s_online;
static int                s_retries;

/* ================================================================== */
/*  Wi-Fi                                                              */
/* ================================================================== */

static void on_wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_online = false;
        if (++s_retries <= 20 || s_retries % 20 == 0) {
            ESP_LOGW(TAG, "Wi-Fi dropped, reconnecting (%d)", s_retries);
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "online as " IPSTR, IP2STR(&e->ip_info.ip));
        s_retries = 0;
        s_online  = true;
        xEventGroupSetBits(s_events, WIFI_CONNECTED_BIT);
    }
}

esp_err_t net_wifi_start(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    s_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t ic = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&ic));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                        on_wifi_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                        on_wifi_event, NULL, NULL));

    wifi_config_t wc = {0};
    strncpy((char *)wc.sta.ssid,     WIFI_SSID,     sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, WIFI_PASSWORD, sizeof(wc.sta.password) - 1);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));   // power save adds latency

    ESP_LOGI(TAG, "joining \"%s\"...", WIFI_SSID);
    EventBits_t bits = xEventGroupWaitBits(s_events, WIFI_CONNECTED_BIT,
                                           pdFALSE, pdTRUE, pdMS_TO_TICKS(30000));
    return (bits & WIFI_CONNECTED_BIT) ? ESP_OK : ESP_ERR_TIMEOUT;
}

bool net_is_online(void) { return s_online; }

/* ================================================================== */
/*  Finding the laptop: UDP broadcast, so no IP to hardcode            */
/* ================================================================== */

esp_err_t net_find_laptop(int timeout_ms, char *ip, size_t ip_len, int *port)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        return ESP_FAIL;
    }

    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &yes, sizeof(yes));
    struct timeval tv = { .tv_sec = 0, .tv_usec = 400 * 1000 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in to = {
        .sin_family      = AF_INET,
        .sin_port        = htons(DISCOVERY_PORT),
        .sin_addr.s_addr = htonl(INADDR_BROADCAST),
    };

    esp_err_t result = ESP_ERR_TIMEOUT;
    for (int waited = 0; waited < timeout_ms; waited += 500) {
        sendto(sock, "I2SONIC?", 8, 0, (struct sockaddr *)&to, sizeof(to));

        char buf[32];
        struct sockaddr_in from;
        socklen_t from_len = sizeof(from);
        int n = recvfrom(sock, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&from, &from_len);
        if (n >= 8 && memcmp(buf, "I2SONIC!", 8) == 0) {
            buf[n] = '\0';
            int p = atoi(buf + 8);                    // friday-local adds its HTTP port
            *port = (p > 0 && p < 65536) ? p : LAPTOP_HTTP_PORT;
            inet_ntoa_r(from.sin_addr, ip, ip_len);
            result = ESP_OK;
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    close(sock);
    return result;
}

bool net_can_connect(const char *ip, int port, int timeout_ms)
{
    struct sockaddr_in to = {
        .sin_family = AF_INET,
        .sin_port   = htons(port),
    };
    if (inet_pton(AF_INET, ip, &to.sin_addr) != 1) {
        return false;
    }
    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
        return false;
    }
    fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) | O_NONBLOCK);

    bool ok = false;
    if (connect(sock, (struct sockaddr *)&to, sizeof(to)) == 0) {
        ok = true;
    } else if (errno == EINPROGRESS) {
        fd_set w;
        FD_ZERO(&w);
        FD_SET(sock, &w);
        struct timeval tv = { .tv_sec = timeout_ms / 1000, .tv_usec = (timeout_ms % 1000) * 1000 };
        if (select(sock + 1, NULL, &w, NULL, &tv) > 0) {
            int so_err = 0;
            socklen_t len = sizeof(so_err);
            getsockopt(sock, SOL_SOCKET, SO_ERROR, &so_err, &len);
            ok = so_err == 0;
        }
    }
    close(sock);
    return ok;
}
