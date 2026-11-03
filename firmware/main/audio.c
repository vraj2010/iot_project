#include "audio.h"

#include <math.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"

#include "config.h"

#if MIC_TYPE == MIC_MAX4466
#include "esp_adc/adc_continuous.h"
#elif MIC_TYPE != MIC_INMP441
#error "MIC_TYPE must be MIC_MAX4466 or MIC_INMP441"
#endif

static const char *TAG = "audio";

/* ================================================================== */
/*  Speaker — MAX98357A on I2S1                                        */
/* ================================================================== */

static i2s_chan_handle_t s_tx;
static bool              s_amp_on;

static inline int16_t clamp16(float v)
{
    if (v > 32767.0f)  return 32767;
    if (v < -32768.0f) return -32768;
    return (int16_t)v;
}

static void speaker_init(void)
{
    gpio_config_t sd = {
        .pin_bit_mask = 1ULL << AMP_SD_GPIO,
        .mode         = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&sd));
    gpio_set_level(AMP_SD_GPIO, 0);          // amp off until we actually play

    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    // 8 x 512 frames = 256 ms of audio queued in hardware, so a busy moment
    // for Wi-Fi or TLS can't starve the speaker (was 6 x 256 = 96 ms).
    cc.dma_desc_num  = 8;
    cc.dma_frame_num = 512;
    cc.auto_clear    = true;                 // silence, not stale audio, on underrun
    ESP_ERROR_CHECK(i2s_new_channel(&cc, &s_tx, NULL));

    i2s_std_config_t sc = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = AMP_BCLK_GPIO,
            .ws   = AMP_LRC_GPIO,
            .dout = AMP_DIN_GPIO,
            .din  = I2S_GPIO_UNUSED,
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_tx, &sc));
    ESP_ERROR_CHECK(i2s_channel_enable(s_tx));
}

static void write_stereo(const int16_t *mono, size_t n)
{
    static int16_t stereo[AUDIO_BLOCK * 2];
    while (n > 0) {
        size_t chunk = n > AUDIO_BLOCK ? AUDIO_BLOCK : n;
        for (size_t i = 0; i < chunk; i++) {
            int16_t s = clamp16((float)mono[i] * VOLUME);
            stereo[2 * i]     = s;           // same sample both channels, so the
            stereo[2 * i + 1] = s;           // amp's L/R jumper doesn't matter
        }
        size_t written = 0;
        i2s_channel_write(s_tx, stereo, chunk * 2 * sizeof(int16_t), &written,
                          portMAX_DELAY);
        mono += chunk;
        n    -= chunk;
    }
}

static void write_silence(size_t samples)
{
    static const int16_t quiet[AUDIO_BLOCK] = {0};
    while (samples > 0) {
        size_t chunk = samples > AUDIO_BLOCK ? AUDIO_BLOCK : samples;
        write_stereo(quiet, chunk);
        samples -= chunk;
    }
}

void audio_play_begin(void)
{
    if (s_amp_on) {
        return;
    }
    write_silence(AUDIO_BLOCK * 2);          // let the I2S clocks settle
    gpio_set_level(AMP_SD_GPIO, 1);
    vTaskDelay(pdMS_TO_TICKS(15));
    s_amp_on = true;
}

void audio_play_chunk(const int16_t *pcm, size_t samples)
{
    write_stereo(pcm, samples);
}

void audio_play_end(void)
{
    if (!s_amp_on) {
        return;
    }
    write_silence(AUDIO_BLOCK * 2);          // trail off before muting, no pop
    gpio_set_level(AMP_SD_GPIO, 0);
    s_amp_on = false;
}

static void (*s_tone_sink)(const int16_t *pcm, size_t samples);

void audio_set_tone_sink(void (*sink)(const int16_t *pcm, size_t samples))
{
    s_tone_sink = sink;
}

void audio_tone(float freq_hz, int ms, float amplitude)
{
    int16_t buf[AUDIO_BLOCK];                // on the caller's stack: tones can come from any task
    int total = SAMPLE_RATE_HZ * ms / 1000;
    int k = 0;

    if (!s_tone_sink) {
        audio_play_begin();
    }
    while (k < total) {
        int n = (total - k) > AUDIO_BLOCK ? AUDIO_BLOCK : (total - k);
        for (int i = 0; i < n; i++, k++) {
            // fade the first and last 5 ms so the tone does not click
            float fade = 1.0f;
            int edge = SAMPLE_RATE_HZ / 200;
            if (k < edge)             fade = (float)k / edge;
            else if (total - k < edge) fade = (float)(total - k) / edge;
            buf[i] = clamp16(32767.0f * amplitude * fade *
                             sinf(2.0f * (float)M_PI * freq_hz * k / SAMPLE_RATE_HZ));
        }
        if (s_tone_sink) {
            s_tone_sink(buf, (size_t)n);
        } else {
            write_stereo(buf, n);
        }
    }
    if (!s_tone_sink) {
        audio_play_end();
    }
}

void audio_chime_start(void) { audio_tone(880.0f,  90, 0.35f); }
void audio_chime_done(void)  { audio_tone(1320.0f, 70, 0.25f); }
void audio_chime_error(void) { audio_tone(220.0f, 180, 0.35f); }

/* ================================================================== */
/*  Microphone                                                         */
/*  mic_raw() delivers AUDIO_BLOCK samples on a 16-bit scale, DC and    */
/*  all. The cleanup chain below turns that into usable speech.         */
/* ================================================================== */

static volatile int s_glitches;

#if MIC_TYPE == MIC_MAX4466

static adc_continuous_handle_t s_adc;
static adc_channel_t           s_adc_ch;
static uint8_t                 s_adc_buf[AUDIO_BLOCK * SOC_ADC_DIGI_RESULT_BYTES];
static adc_continuous_data_t   s_parsed[AUDIO_BLOCK];

static inline uint32_t median3(uint32_t a, uint32_t b, uint32_t c)
{
    if (a > b) { uint32_t t = a; a = b; b = t; }
    if (b > c) { b = c; }
    return a > b ? a : b;
}

static void mic_init(void)
{
    adc_unit_t unit;
    ESP_ERROR_CHECK(adc_continuous_io_to_channel(MAX4466_OUT_GPIO, &unit, &s_adc_ch));
    if (unit != ADC_UNIT_1) {
        ESP_LOGE(TAG, "GPIO %d is not ADC1. Use GPIO 1-10 (ADC2 clashes with Wi-Fi).",
                 MAX4466_OUT_GPIO);
        abort();
    }

    adc_continuous_handle_cfg_t hc = {
        .max_store_buf_size = AUDIO_BLOCK * SOC_ADC_DIGI_RESULT_BYTES * 8,
        .conv_frame_size    = AUDIO_BLOCK * SOC_ADC_DIGI_RESULT_BYTES,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&hc, &s_adc));

    adc_digi_pattern_config_t pattern = {
        .atten     = ADC_ATTEN_DB_12,            // ~0-3.1 V; the mic idles near 1.65 V
        .channel   = s_adc_ch,
        .unit      = ADC_UNIT_1,
        .bit_width = SOC_ADC_DIGI_MAX_BITWIDTH,  // 12-bit
    };
    adc_continuous_config_t cfg = {
        .pattern_num    = 1,
        .adc_pattern    = &pattern,
        .sample_freq_hz = SAMPLE_RATE_HZ,
        .conv_mode      = ADC_CONV_SINGLE_UNIT_1,
        .format         = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    };
    ESP_ERROR_CHECK(adc_continuous_config(s_adc, &cfg));
    ESP_ERROR_CHECK(adc_continuous_start(s_adc));
}

static void mic_raw(float *out)
{
    static uint32_t last_good = 2048, h1 = 2048, h2 = 2048;
    size_t got = 0;

    while (got < AUDIO_BLOCK) {
        uint32_t len = 0;
        esp_err_t err = adc_continuous_read(
            s_adc, s_adc_buf, (AUDIO_BLOCK - got) * SOC_ADC_DIGI_RESULT_BYTES,
            &len, 1000);
        if (err != ESP_OK) {
            continue;
        }
        uint32_t count = 0;
        if (adc_continuous_parse_data(s_adc, s_adc_buf, len, s_parsed, &count) != ESP_OK) {
            continue;
        }
        for (uint32_t i = 0; i < count && got < AUDIO_BLOCK; i++) {
            const adc_continuous_data_t *d = &s_parsed[i];
            uint32_t raw = d->raw_data;
            // A reading that is invalid, from the wrong channel, or a bare 0 is a
            // DMA glitch rather than audio — hold the previous value instead.
            if (!d->valid || d->unit != ADC_UNIT_1 || d->channel != s_adc_ch || raw == 0) {
                raw = last_good;
                s_glitches++;
            } else {
                last_good = raw;
            }
            uint32_t m = median3(h2, h1, raw);   // kills any single-sample spike
            h2 = h1;
            h1 = raw;
            out[got++] = (float)(m << 4);        // 12-bit -> 16-bit scale
        }
    }
}

void audio_flush_mic(void)
{
    uint32_t len;
    for (int i = 0; i < 16; i++) {
        if (adc_continuous_read(s_adc, s_adc_buf, sizeof(s_adc_buf), &len, 0) != ESP_OK) {
            break;
        }
    }
}

#else   /* ---------------- INMP441 on I2S0 ---------------- */

static i2s_chan_handle_t s_rx;
static int32_t           s_i2s_buf[AUDIO_BLOCK];

static void mic_init(void)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num  = 6;
    cc.dma_frame_num = AUDIO_BLOCK;
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &s_rx));

    i2s_std_config_t sc = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = INMP441_SCK_GPIO,
            .ws   = INMP441_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = INMP441_SD_GPIO,
        },
    };
    sc.slot_cfg.slot_mask = INMP441_RIGHT_SLOT ? I2S_STD_SLOT_RIGHT : I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &sc));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
}

static void mic_raw(float *out)
{
    size_t bytes = 0;
    i2s_channel_read(s_rx, s_i2s_buf, sizeof(s_i2s_buf), &bytes, portMAX_DELAY);
    size_t got = bytes / sizeof(int32_t);
    for (size_t i = 0; i < AUDIO_BLOCK; i++) {
        int32_t w = (i < got) ? s_i2s_buf[i] : 0;
        out[i] = (float)(w >> 16) * 4.0f;    // 24-in-32 -> 16-bit, +12 dB
    }
}

void audio_flush_mic(void)
{
    size_t bytes;
    for (int i = 0; i < 16; i++) {
        if (i2s_channel_read(s_rx, s_i2s_buf, sizeof(s_i2s_buf), &bytes, 0) != ESP_OK) {
            break;
        }
    }
}

#endif

/* ================================================================== */
/*  Cleanup chain: DC removal -> high-pass -> noise gate               */
/* ================================================================== */

static float s_hp_x1, s_hp_y;     // one-pole high-pass state
static float s_dc = 2048.0f * 16; // slow DC estimate
static float s_env;               // envelope follower
static float s_gate = 1.0f;       // smoothed gate gain
static float s_level_db = -90.0f;

#define HP_A      (expf(-2.0f * (float)M_PI * HPF_CUTOFF_HZ / SAMPLE_RATE_HZ))
#define DC_ALPHA  0.0005f
#define ENV_ATK   0.30f
#define ENV_REL   0.0020f

void audio_capture(int16_t *out)
{
    static float raw[AUDIO_BLOCK];
    const float gate_thresh = powf(10.0f, NOISE_GATE_DBFS / 20.0f) * 32768.0f;

    s_glitches = 0;
    mic_raw(raw);

    double sumsq = 0;
    for (int i = 0; i < AUDIO_BLOCK; i++) {
        float x = raw[i];

        // 1. track and subtract the DC offset (the MAX4466 idles around mid-rail)
        s_dc += (x - s_dc) * DC_ALPHA;
        x -= s_dc;

        // 2. one-pole high-pass: removes mains hum and breath rumble
        s_hp_y  = HP_A * (s_hp_y + x - s_hp_x1);
        s_hp_x1 = x;
        x = s_hp_y;

        // 3. noise gate: duck the hiss between words so Whisper does not
        //    invent text for the silence
        float a = fabsf(x);
        s_env += (a - s_env) * (a > s_env ? ENV_ATK : ENV_REL);
        float target = (s_env > gate_thresh) ? 1.0f : GATE_FLOOR;
        s_gate += (target - s_gate) * 0.05f;
        x *= s_gate;

        out[i] = clamp16(x);
        sumsq += (double)out[i] * out[i];
    }

    float rms = sqrtf((float)(sumsq / AUDIO_BLOCK));
    s_level_db = rms > 0.5f ? 20.0f * log10f(rms / 32768.0f) : -90.0f;
}

float audio_level_db(void)  { return s_level_db; }
int   audio_glitch_count(void) { return s_glitches; }

float audio_normalize(int16_t *pcm, size_t samples)
{
    int32_t peak = 0;
    for (size_t i = 0; i < samples; i++) {
        int32_t a = pcm[i] < 0 ? -(int32_t)pcm[i] : pcm[i];
        if (a > peak) {
            peak = a;
        }
    }
    if (peak < 64) {                 // essentially silence, leave it alone
        return 1.0f;
    }
    // Gain below 1.0 is fine and wanted: it pulls a shouted question back down.
    float gain = (NORMALIZE_PEAK * 32767.0f) / (float)peak;
    if (gain > NORMALIZE_MAX_X) {
        gain = NORMALIZE_MAX_X;
    }
    for (size_t i = 0; i < samples; i++) {
        pcm[i] = clamp16((float)pcm[i] * gain);
    }
    return gain;
}

/* ================================================================== */

void audio_init(void)
{
    speaker_init();
    mic_init();

    // Seed the DC estimate from real samples so the first block is not a thump
    static float raw[AUDIO_BLOCK];
    mic_raw(raw);
    float sum = 0;
    for (int i = 0; i < AUDIO_BLOCK; i++) {
        sum += raw[i];
    }
    s_dc = sum / AUDIO_BLOCK;

    ESP_LOGI(TAG, "ready (%s mic, %d Hz)",
             MIC_TYPE == MIC_MAX4466 ? "MAX4466" : "INMP441", SAMPLE_RATE_HZ);
}
