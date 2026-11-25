// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 AI-FanGe
//
// Head board: face, speaker and camera for Xiaoxiaobai.
// The Pi streams 64x48 LED masks and tone requests over a 3-wire UART (USB CDC also
// accepted, see src/esp32_link.py); this sketch renders the LED-matrix face on the
// 2 inch LCD, drives an I2S amplifier (tones or streamed PCM) and, while the controller asks for
// them, streams the Sense camera as JPEG and the Sense PDM microphone as PCM back over the
// same UART.
// The Pi also hands over its saved Wi-Fi networks: once joined, the board serves the camera
// at http://xiaoxiaobai-face.local/ and accepts OTA firmware updates (password kOtaPassword).
// With a direct voice config from the Pi (API key kept in RAM only), the board itself holds
// the Qwen realtime WebSocket sessions: microphone, camera and reply audio never cross the UART.
// Arduino IDE: board XIAO_ESP32S3, Tools > PSRAM > "OPI PSRAM" (camera frame buffers).

// Bluetooth LE link (XIAOXIAOBAI_DISPLAY_BACKEND=ble). Off by default: with Wi-Fi on, its RAM
// and radio time are missing for the camera stream.
#ifndef XIAOXIAOBAI_BLE
#define XIAOXIAOBAI_BLE 0
#endif

#include <ArduinoOTA.h>
#include <Update.h>
#include <esp_task_wdt.h>
#include <ESPmDNS.h>
#include <ESP_I2S.h>
#include <Preferences.h>
#include <driver/ledc.h>
#include <SPI.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <cJSON.h>
#include <esp_crt_bundle.h>
#include <esp_http_server.h>
#include <esp_transport.h>
#include <esp_transport_ssl.h>
#include <esp_transport_tcp.h>
#include <esp_transport_ws.h>
#include <freertos/stream_buffer.h>
#include <mbedtls/base64.h>
#include <mbedtls/platform.h>
#if XIAOXIAOBAI_BLE
#include <BLEDevice.h>
#include <BLEServer.h>
#endif

#include <atomic>
#include <initializer_list>
#include <memory>
#include <string>

#include "face_render.h"
#include "protocol.h"
#include "voice_session.h"

// Pi GPIO14 (TXD) -> D9, Pi GPIO15 (RXD) <- D2. Must match ESP32_UART_BAUD in src/esp32_link.py.
constexpr int kLinkRx = D9;
constexpr int kLinkTx = D2;
constexpr uint32_t kLinkBaud = 3000000;
// Room for the Pi's PCM bursts while loop() is busy pushing an LCD frame (~30 ms).
constexpr size_t kLinkRxBufferBytes = 16384;
constexpr uint32_t kStatusIntervalMs = 1000;

// The Pi resends the camera request every second while someone is watching.
constexpr uint32_t kCameraKeepaliveMs = 3000;
constexpr uint32_t kCameraRetryMs = 3000;
constexpr size_t kCameraChunkBytes = 4000;
// Arduino's Serial*.write() holds the port mutex until the bytes are in the FIFO; small
// slices keep the main loop's reads of the same port responsive.
constexpr size_t kLinkWriteSlice = 256;

constexpr char kHostname[] = "xiaoxiaobai-face";  // mDNS name, also the Arduino IDE network port
constexpr char kOtaPassword[] = "xiaoxiaobai";    // ESP32_OTA_PASSWORD in the Makefile
constexpr uint16_t kWebPort = 80;
constexpr uint16_t kStreamPort = 81;
constexpr uint32_t kWifiRetryMs = 5000;
constexpr uint32_t kWifiJoinTimeoutMs = 15000;
constexpr uint32_t kStreamIdleMs = 10000;
constexpr size_t kSharedJpegBytes = 128 * 1024;

#if XIAOXIAOBAI_BLE
constexpr char kBleName[] = "xiaoxiaobai-face";
constexpr char kBleServiceUuid[] = "8f6a0001-3c1e-4d2a-9b7e-6d6963726f62";
constexpr char kBleRxUuid[] = "8f6a0002-3c1e-4d2a-9b7e-6d6963726f62";
// The Pi sends a face every 80 ms; a link silent for this long belongs to a dead client.
constexpr uint32_t kBleIdleDisconnectMs = 5000;
constexpr uint16_t kNoConnection = 0xFFFF;
#endif

constexpr int kLcdCs = D0;
constexpr int kLcdDc = D1;
constexpr int kLcdRst = D3;
constexpr int kLcdBl = D4;
constexpr int kLcdSck = D8;
constexpr int kLcdMosi = D10;
constexpr uint32_t kLcdSpiHz = 40000000;
// Backlight PWM straight on the IDF driver: the camera's XCLK takes LEDC timer 0 / channel 0 behind
// the Arduino LEDC allocator's back, which would otherwise hand those to the backlight as well.
constexpr ledc_timer_t kLcdBlTimer = LEDC_TIMER_1;
constexpr ledc_channel_t kLcdBlChannel = LEDC_CHANNEL_1;
constexpr ledc_timer_bit_t kLcdBlBits = LEDC_TIMER_10_BIT;
constexpr uint32_t kLcdBlHz = 20000;  // above the audio band
constexpr uint8_t kBrightnessDefault = 100;
constexpr int kPanelWidth = face::kHeight;
constexpr int kPanelHeight = face::kWidth;

constexpr int kI2sBclk = D5;
constexpr int kI2sLrc = D6;
constexpr int kI2sDout = D7;
constexpr uint32_t kSampleRate = protocol::kPcmSampleRate;
// Speaker samples queued for playback. The Pi paces its sends ~100 ms ahead of playback; a
// direct voice session queues whole replies as fast as Qwen sends them (flushed on barge-in).
constexpr size_t kPcmBufferBytes = protocol::kPcmSampleRate * 2 * 10;
constexpr size_t kPcmBufferInternalBytes = protocol::kPcmSampleRate * 2 / 4;
// Speech is heard this long after its loudness goes to the Pi, which needs the head start to
// move the face's mouth in time (report, frame, UART, render).
constexpr uint32_t kSpeechDelayMs = 20;
constexpr size_t kSpeechDelaySamples = kSampleRate * kSpeechDelayMs / 1000;
// The PCM stream ran dry this long: the reply paused or ended, so the delayed tail is played.
// Shorter gaps are covered by the I2S DMA and keep the delay line as it is.
constexpr uint32_t kSpeechDrainMs = 30;

// Sense expansion board PDM microphone. PDM RX only exists on I2S0.
constexpr int kMicClk = 42;
constexpr int kMicData = 41;
constexpr size_t kMicChunkSamples = protocol::kMicSampleRate / 50;  // 20 ms
// The Pi resends the microphone request every second while a voice client is attached.
constexpr uint32_t kMicKeepaliveMs = 3000;

constexpr uint32_t kHeapLogIntervalMs = 10000;

// The loading spinner runs from power-on until the Pi sends a face, and again if the faces stop
// for kFaceStaleMs while the backlight is on (Pi rebooting); spinner and face crossfade.
constexpr uint32_t kLoadingFrameMs = 40;
constexpr uint32_t kFaceFadeMs = 500;
constexpr uint32_t kFaceStaleMs = 5000;
// kMsgImage: worst case of unpackPixelRle (all literals) for one panel frame; longest it may stay up.
constexpr size_t kImagePixels = size_t(kPanelWidth) * kPanelHeight;
constexpr size_t kImageMaxBytes = (kImagePixels + 127) / 128 * 257;
constexpr uint32_t kImageMaxShowMs = 10 * 60 * 1000;

// Direct voice session (ESP32 <-> Qwen realtime over TLS WebSocket).
constexpr uint32_t kVoiceKeepaliveMs = 5000;
constexpr int kVoiceConnectTimeoutMs = 10000;
constexpr int kVoiceIoTimeoutMs = 2000;
constexpr uint32_t kVoiceRetryMs = 3000;
// The microphone stays muted this long after the last PCM went to the DMA (no echo cancellation).
constexpr uint32_t kVoiceEchoTailMs = 350;
constexpr uint32_t kVoiceStatsIntervalMs = 3000;
// The camera stays off until the live transcript asks for a look, then runs only until that
// one frame is sent or the user stops talking.
constexpr size_t kVoiceMaxConfigBytes = 8192;
constexpr size_t kVoiceMaxContextBytes = 16384;
constexpr size_t kVoiceMaxMessageBytes = 512 * 1024;
constexpr size_t kVoiceMicChunkBytes = kMicChunkSamples * sizeof(int16_t);
constexpr size_t kVoiceMicBufferBytes = protocol::kMicSampleRate * 2;  // 1 s
constexpr size_t kVoiceMicBatchChunks = 5;    // up to 100 ms of audio per message
constexpr int kVoiceMicMessagesPerPump = 2;
// The first clause of a reply goes to the TTS on its own to start speaking sooner (UTF-8 bytes).
constexpr size_t kTtsFirstClauseBytes = 6;

struct Tone {
  uint16_t freq_hz;
  uint16_t duration_ms;
  uint8_t volume;
};

SPIClass lcd_spi(FSPI);
I2SClass mic(I2S_NUM_0);
I2SClass i2s(I2S_NUM_1);
QueueHandle_t tone_queue = nullptr;
StreamBufferHandle_t pcm_stream = nullptr;
StaticStreamBuffer_t pcm_stream_struct;
std::atomic<bool> pcm_flush{false};
// millis() of the last PCM handed to the I2S DMA (echo gating of the direct voice session).
volatile uint32_t last_pcm_ms = 0;
volatile bool mic_on = false;
volatile bool mic_failed = false;
volatile bool mic_wanted = false;
volatile uint32_t mic_request_ms = 0;
Print* volatile mic_out = nullptr;
face::Renderer renderer;
// With PSRAM there are two framebuffers: loop() renders the next face into back_framebuffer
// while lcdTask pushes `framebuffer` to the panel.
uint16_t* framebuffer = nullptr;
uint16_t* back_framebuffer = nullptr;
TaskHandle_t lcd_task = nullptr;
std::atomic<bool> lcd_busy{false};
uint8_t face_levels[face::kCells];
uint8_t face_levels_next[face::kCells];  // kMsgFaceLevels bands land here until the frame is committed
bool face_dirty = false;
bool face_live = false;      // the Pi is sending faces
uint32_t face_last_ms = 0;   // millis() of its last one
float face_mix = 0.0f;       // 0 spinner .. 1 face
uint8_t face_shown[face::kCells];  // the two crossfaded
uint32_t face_frames = 0;  // render stats for the heap log
uint32_t face_render_us = 0;
uint32_t face_render_max_us = 0;
Print* volatile face_out = nullptr;  // port the face comes from; speaker levels go back there
// kMsgImage: packed bytes as they arrive, the decoded frame, and while it is up, until when.
uint8_t* image_packed = nullptr;
uint16_t* image_frame = nullptr;
uint32_t image_size = 0;
uint32_t image_received = 0;
uint32_t image_crc = 0;
uint32_t image_show_ms = 0;
uint32_t image_until_ms = 0;
bool image_active = false;
bool image_dirty = false;
// seq u16 << 16 | loudness u8 << 8 | zero-crossing rate u8 of the last speaker measurement (kMsgSpeaker).
std::atomic<uint32_t> speaker_report{0};
bool backlight_wanted = true;
uint8_t brightness_wanted = kBrightnessDefault;  // % (kMsgBrightness)
uint8_t backlight_level = 0;                     // % on the pin, 0 while the backlight is off

#if XIAOXIAOBAI_BLE
BLEServer* ble_server = nullptr;
StreamBufferHandle_t ble_rx = nullptr;
volatile uint16_t ble_conn = kNoConnection;
volatile uint32_t ble_last_rx_ms = 0;
#endif

HardwareSerial& link_uart = Serial1;
// Held for a whole Pi-bound packet: the camera and microphone tasks both write.
SemaphoreHandle_t link_tx_lock = nullptr;
// Port that asked for the camera (UART or USB); frames go back the same way.
Print* volatile camera_out = nullptr;
volatile bool camera_wanted = false;
volatile uint32_t camera_request_ms = 0;
volatile uint8_t camera_size_code = 1;
volatile uint8_t camera_quality = 12;
volatile uint32_t camera_interval_ms = 0;
volatile bool camera_on = false;
volatile bool camera_failed = false;

// Latest frame for the Wi-Fi viewers, copied out of the camera buffer (PSRAM).
SemaphoreHandle_t jpeg_lock = nullptr;
uint8_t* shared_jpeg = nullptr;
size_t shared_jpeg_size = 0;
std::atomic<uint32_t> shared_jpeg_seq{0};
volatile uint32_t shared_jpeg_ms = 0;
// Direct voice session: microphone routing and camera frames for the model.
std::atomic<bool> voice_active{false};
StreamBufferHandle_t voice_mic_stream = nullptr;
volatile bool voice_camera_wanted = false;
std::atomic<bool> voice_frame_request{false};  // the voice session wants one fresh frame in shared_jpeg
std::atomic<uint32_t> voice_mic_dropped{0};    // 20 ms chunks lost because voice_mic_stream was full
volatile uint8_t voice_size_code = 3;
volatile uint8_t voice_quality = 14;
std::atomic<int> wifi_viewers{0};
volatile uint8_t wifi_size_code = 3;
volatile uint8_t wifi_quality = 12;

// Wi-Fi networks pushed by the Pi (kMsgWifi payload), persisted in NVS.
struct WifiNetwork {
  char ssid[33];
  char psk[65];
};
Preferences prefs;
SemaphoreHandle_t wifi_config_lock = nullptr;
uint8_t wifi_config[protocol::kMaxPayload];
uint16_t wifi_config_size = 0;
volatile uint32_t wifi_config_crc = 0;
volatile bool wifi_config_dirty = false;
volatile bool ota_active = false;
// UART OTA in progress: the camera, the voice session and Wi-Fi OTA stay out of its way.
volatile bool uart_ota = false;
volatile uint8_t wifi_state = protocol::kWifiNoConfig;
volatile uint8_t wifi_seen = 0;
volatile int8_t wifi_saved_rssi = 0;
volatile uint8_t wifi_reason = 0;

// ---------------------------------------------------------------- LCD

void lcdCommand(uint8_t value) {
  digitalWrite(kLcdDc, LOW);
  digitalWrite(kLcdCs, LOW);
  lcd_spi.write(value);
  digitalWrite(kLcdCs, HIGH);
}

void lcdData(const uint8_t* data, size_t size) {
  digitalWrite(kLcdDc, HIGH);
  digitalWrite(kLcdCs, LOW);
  lcd_spi.writeBytes(data, size);
  digitalWrite(kLcdCs, HIGH);
}

void lcdCommand(uint8_t value, std::initializer_list<uint8_t> data) {
  lcdCommand(value);
  lcdData(data.begin(), data.size());
}

void lcdSetWindow(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
  lcdCommand(0x2A, {uint8_t(x0 >> 8), uint8_t(x0), uint8_t(x1 >> 8), uint8_t(x1)});
  lcdCommand(0x2B, {uint8_t(y0 >> 8), uint8_t(y0), uint8_t(y1 >> 8), uint8_t(y1)});
  lcdCommand(0x2C);
}

void lcdPushFrame(const uint16_t* frame) {
  lcd_spi.beginTransaction(SPISettings(kLcdSpiHz, MSBFIRST, SPI_MODE0));
  lcdSetWindow(0, 0, kPanelWidth - 1, kPanelHeight - 1);
  lcdData(reinterpret_cast<const uint8_t*>(frame), size_t(kPanelWidth) * kPanelHeight * 2);
  lcd_spi.endTransaction();
}

void lcdTask(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    lcdPushFrame(framebuffer);
    lcd_busy = false;
  }
}

// Same register sequence as SpiLcd.init() in src/display_smile.py.
void lcdInit() {
  pinMode(kLcdCs, OUTPUT);
  pinMode(kLcdDc, OUTPUT);
  pinMode(kLcdRst, OUTPUT);
  digitalWrite(kLcdCs, HIGH);
  ledc_timer_config_t bl_timer = {};
  bl_timer.speed_mode = LEDC_LOW_SPEED_MODE;
  bl_timer.duty_resolution = kLcdBlBits;
  bl_timer.timer_num = kLcdBlTimer;
  bl_timer.freq_hz = kLcdBlHz;
  bl_timer.clk_cfg = LEDC_AUTO_CLK;  // same as the camera's XCLK: all low-speed timers share one source
  ledc_timer_config(&bl_timer);
  ledc_channel_config_t bl_channel = {};
  bl_channel.gpio_num = kLcdBl;
  bl_channel.speed_mode = LEDC_LOW_SPEED_MODE;
  bl_channel.channel = kLcdBlChannel;
  bl_channel.timer_sel = kLcdBlTimer;
  bl_channel.duty = 0;
  ledc_channel_config(&bl_channel);
  lcd_spi.begin(kLcdSck, -1, kLcdMosi, -1);

  digitalWrite(kLcdRst, HIGH);
  delay(10);
  digitalWrite(kLcdRst, LOW);
  delay(10);
  digitalWrite(kLcdRst, HIGH);
  delay(120);

  lcd_spi.beginTransaction(SPISettings(kLcdSpiHz, MSBFIRST, SPI_MODE0));
  lcdCommand(0x36, {0x00});
  lcdCommand(0x3A, {0x55});
  lcdCommand(0x21);
  lcdCommand(0x2A, {0x00, 0x00, 0x00, 0xEF});
  lcdCommand(0x2B, {0x00, 0x00, 0x01, 0x3F});
  lcdCommand(0xB2, {0x0C, 0x0C, 0x00, 0x33, 0x33});
  lcdCommand(0xB7, {0x35});
  lcdCommand(0xBB, {0x1F});
  lcdCommand(0xC0, {0x2C});
  lcdCommand(0xC2, {0x01});
  lcdCommand(0xC3, {0x12});
  lcdCommand(0xC4, {0x20});
  lcdCommand(0xC6, {0x0F});
  lcdCommand(0xD0, {0xA4, 0xA1});
  lcdCommand(0xE0, {0xD0, 0x08, 0x11, 0x08, 0x0C, 0x15, 0x39, 0x33, 0x50, 0x36, 0x13, 0x14, 0x29, 0x2D});
  lcdCommand(0xE1, {0xD0, 0x08, 0x10, 0x08, 0x06, 0x06, 0x39, 0x44, 0x51, 0x0B, 0x16, 0x14, 0x2F, 0x31});
  lcdCommand(0x11);
  lcd_spi.endTransaction();
  delay(120);
  lcd_spi.beginTransaction(SPISettings(kLcdSpiHz, MSBFIRST, SPI_MODE0));
  lcdCommand(0x29);
  lcd_spi.endTransaction();

  memset(framebuffer, 0, size_t(kPanelWidth) * kPanelHeight * 2);
  lcdPushFrame(framebuffer);
}

// Squared, so equal slider steps look like equal steps in brightness. A duty of 1 << bits is fully on.
void applyBacklight() {
  const uint8_t level = backlight_wanted ? brightness_wanted : 0;
  if (level == backlight_level) return;
  backlight_level = level;
  const float share = level / 100.0f;
  const uint32_t duty = level == 0 ? 0 : max<uint32_t>(1, lrintf(share * share * (1u << kLcdBlBits)));
  ledc_set_duty(LEDC_LOW_SPEED_MODE, kLcdBlChannel, duty);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, kLcdBlChannel);
}

// ---------------------------------------------------------------- audio

constexpr size_t kAudioChunkFrames = 256;

void playTone(const Tone& tone, int16_t* buffer) {
  constexpr uint32_t kRampSamples = kSampleRate * 5 / 1000;
  const uint32_t total = kSampleRate * tone.duration_ms / 1000;
  const float amplitude = 32767.0f * 0.6f * min<uint8_t>(tone.volume, 100) / 100.0f;
  const float step = 2.0f * PI * tone.freq_hz / kSampleRate;
  float phase = 0.0f;
  for (uint32_t done = 0; done < total;) {
    const size_t frames = min<uint32_t>(kAudioChunkFrames, total - done);
    for (size_t i = 0; i < frames; ++i, ++done) {
      const uint32_t edge = min(done, total - done);
      const float envelope = edge < kRampSamples ? float(edge) / kRampSamples : 1.0f;
      const int16_t sample = tone.freq_hz ? int16_t(sinf(phase) * amplitude * envelope) : 0;
      buffer[2 * i] = sample;
      buffer[2 * i + 1] = sample;
      phase += step;
      if (phase > 2.0f * PI) phase -= 2.0f * PI;
    }
    i2s.write(reinterpret_cast<uint8_t*>(buffer), frames * 2 * sizeof(int16_t));
  }
}

void speakerPublish(uint8_t loudness, uint8_t zcr) {
  const uint32_t seq = ((speaker_report.load() >> 16) + 1) & 0xFFFF;
  speaker_report = seq << 16 | uint32_t(loudness) << 8 | zcr;
}

// Loudness and zero-crossing rate of what goes into the I2S DMA, for the lip-sync on the Pi. Measured as
// the samples are queued, so the report runs a little ahead of the sound, like the face's render delay.
void speakerMeasure(const int16_t* samples, size_t count) {
  constexpr uint32_t kReportSamples = 512;  // ~21 ms
  static uint64_t energy = 0;
  static uint32_t crossings = 0;
  static uint32_t total = 0;
  static int16_t last = 0;
  for (size_t i = 0; i < count; ++i) {
    const int32_t s = samples[i];
    energy += uint64_t(s * s);
    crossings += (s ^ last) < 0;
    last = int16_t(s);
  }
  total += count;
  if (total < kReportSamples) return;
  const float loudness = 20.0f * log10f(sqrtf(float(energy) / total) + 1.0f) * protocol::kSpeakerDbScale;
  speakerPublish(uint8_t(min(loudness, 255.0f)), uint8_t(min<uint32_t>(255, crossings * 255 / total)));
  energy = 0;
  crossings = 0;
  total = 0;
}

int16_t speech_delay[kSpeechDelaySamples];
size_t speech_delay_pos = 0;

// The speech streams peak well under full scale and the amplifier has no volume of its own:
// speech is lifted by kSpeechGain and a peak limiter (instant attack, kSpeechReleaseS release)
// holds it under full scale instead of clipping. The volume setting still scales it before.
// src/esp32_link.py mirrors both (SPEAKER_PCM_*): the music player sends songs pre-divided.
constexpr float kSpeechGain = 2.8f;  // +9 dB
constexpr float kSpeechCeiling = 0.97f * 32767.0f;
constexpr float kSpeechReleaseS = 0.06f;
float speech_peak = 0.0f;

int16_t speechLoud(int16_t sample) {
  static const float release = expf(-1.0f / (kSpeechReleaseS * kSampleRate));
  const float y = sample * kSpeechGain;
  const float level = fabsf(y);
  speech_peak = level > speech_peak ? level : speech_peak * release;
  return int16_t(speech_peak > kSpeechCeiling ? y * (kSpeechCeiling / speech_peak) : y);
}

// Plays the speech held back in the delay line and empties it.
void drainSpeechDelay(int16_t* buffer) {
  for (size_t done = 0; done < kSpeechDelaySamples;) {
    const size_t frames = min(kAudioChunkFrames, kSpeechDelaySamples - done);
    for (size_t i = 0; i < frames; ++i, ++done) {
      buffer[2 * i] = buffer[2 * i + 1] = speechLoud(speech_delay[(speech_delay_pos + done) % kSpeechDelaySamples]);
    }
    i2s.write(reinterpret_cast<uint8_t*>(buffer), frames * 2 * sizeof(int16_t));
  }
  memset(speech_delay, 0, sizeof(speech_delay));
  speech_delay_pos = 0;
}

// Streamed PCM (kMsgPcm) has priority; tones play while no PCM is queued. The I2S DMA
// plays silence on underrun (auto_clear), so a gap in the stream is just a pause.
void audioTask(void*) {
  int16_t buffer[kAudioChunkFrames * 2];
  int16_t mono[kAudioChunkFrames];
  Tone tone;
  bool speaking = false;
  uint32_t dry_since = 0;
  for (;;) {
    if (pcm_stream == nullptr) {
      if (xQueueReceive(tone_queue, &tone, portMAX_DELAY) == pdTRUE) playTone(tone, buffer);
      continue;
    }
    if (pcm_flush.exchange(false)) {
      while (xStreamBufferReceive(pcm_stream, mono, sizeof(mono), 0) > 0) {
      }
      memset(speech_delay, 0, sizeof(speech_delay));
      speech_delay_pos = 0;
    }
    const size_t got = xStreamBufferReceive(pcm_stream, mono, sizeof(mono), pdMS_TO_TICKS(5)) / sizeof(int16_t);
    if (got > 0) {
      speakerMeasure(mono, got);
      for (size_t i = 0; i < got; ++i) {
        const int16_t sample = speech_delay[speech_delay_pos];
        speech_delay[speech_delay_pos] = mono[i];
        speech_delay_pos = (speech_delay_pos + 1) % kSpeechDelaySamples;
        buffer[2 * i] = buffer[2 * i + 1] = speechLoud(sample);
      }
      speaking = true;
      dry_since = millis();
      i2s.write(reinterpret_cast<uint8_t*>(buffer), got * 2 * sizeof(int16_t));
      last_pcm_ms = millis();
      continue;
    }
    if (speaking) {
      if (millis() - dry_since < kSpeechDrainMs) continue;
      drainSpeechDelay(buffer);
      last_pcm_ms = millis();
      speaking = false;
      speakerPublish(0, 0);
    }
    if (xQueueReceive(tone_queue, &tone, 0) == pdTRUE) playTone(tone, buffer);
  }
}

void audioInit() {
  tone_queue = xQueueCreate(32, sizeof(Tone));
  // FreeRTOS wants one spare byte past the stream buffer's capacity.
  size_t pcm_bytes = kPcmBufferBytes;
  uint8_t* storage = static_cast<uint8_t*>(heap_caps_malloc(pcm_bytes + 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (storage == nullptr) {
    pcm_bytes = kPcmBufferInternalBytes;
    storage = static_cast<uint8_t*>(heap_caps_malloc(pcm_bytes + 1, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  }
  if (storage != nullptr) pcm_stream = xStreamBufferCreateStatic(pcm_bytes, sizeof(int16_t), storage, &pcm_stream_struct);
  i2s.setPins(kI2sBclk, kI2sLrc, kI2sDout);
  if (!i2s.begin(I2S_MODE_STD, kSampleRate, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO)) return;
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, nullptr, 2, nullptr, 0);
}

// Must run before audioInit(): the speaker would otherwise take I2S0, the only PDM-capable port.
void micInit() {
  mic.setPinsPdmRx(kMicClk, kMicData);
  mic_on = mic.begin(I2S_MODE_PDM_RX, protocol::kMicSampleRate, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  mic_failed = !mic_on;
  mic.setTimeout(100);
}

// ---------------------------------------------------------------- Pi-bound packets

void linkWrite(Print* out, const uint8_t* data, size_t size) {
  for (size_t done = 0; done < size;) {
    const size_t written = out->write(data + done, min(kLinkWriteSlice, size - done));
    if (written == 0) return;  // USB host gone
    done += written;
  }
}

void linkSend(Print* out, uint8_t type, const uint8_t* head, size_t head_size, const uint8_t* body,
              size_t body_size) {
  uint8_t checksum = 0;
  for (size_t i = 0; i < head_size; ++i) checksum ^= head[i];
  for (size_t i = 0; i < body_size; ++i) checksum ^= body[i];
  const size_t length = head_size + body_size;
  const uint8_t header[] = {protocol::kMagic0, protocol::kMagic1, type, uint8_t(length), uint8_t(length >> 8)};
  xSemaphoreTake(link_tx_lock, portMAX_DELAY);
  linkWrite(out, header, sizeof(header));
  linkWrite(out, head, head_size);
  linkWrite(out, body, body_size);
  linkWrite(out, &checksum, 1);
  xSemaphoreGive(link_tx_lock);
}

void sendStatus() {
  uint8_t status[protocol::kStatusBytes] = {};
  uint8_t& flags = status[0];
  if (framebuffer != nullptr) flags |= protocol::kStatusFramebuffer;
  if (psramFound()) flags |= protocol::kStatusPsram;
  if (camera_on) flags |= protocol::kStatusCameraOn;
  if (camera_failed) flags |= protocol::kStatusCameraFailed;
  if (ota_active) flags |= protocol::kStatusOta;
  if (mic_on) flags |= protocol::kStatusMicOn;
  if (mic_failed) flags |= protocol::kStatusMicFailed;
  if (WiFi.status() == WL_CONNECTED) {
    flags |= protocol::kStatusWifi;
    const uint32_t ip = WiFi.localIP();
    memcpy(status + 1, &ip, 4);
    status[5] = uint8_t(int8_t(WiFi.RSSI()));
  }
  const uint32_t crc = wifi_config_crc;
  memcpy(status + 6, &crc, 4);
  status[10] = wifi_state;
  status[11] = wifi_seen;
  status[12] = uint8_t(wifi_saved_rssi);
  status[13] = wifi_reason;
  status[14] = protocol::kFeatureFaceLevels | protocol::kFeatureImage | protocol::kFeatureBrightness;
  linkSend(&link_uart, protocol::kMsgStatus, status, sizeof(status), nullptr, 0);
}

// ---------------------------------------------------------------- microphone (Sense board)

// Reads the PDM microphone continuously (so the next chunk is always fresh) and forwards
// 20 ms chunks while the Pi keeps asking for them, and to the direct voice session.
void micTask(void*) {
  int16_t samples[kMicChunkSamples];
  for (;;) {
    const size_t got = mic.readBytes(reinterpret_cast<char*>(samples), sizeof(samples));
    Print* out = mic_out;
    const bool wanted = mic_wanted && out != nullptr && millis() - mic_request_ms < kMicKeepaliveMs && !ota_active;
    if (got > 0 && wanted) {
      linkSend(out, protocol::kMsgMicChunk, nullptr, 0, reinterpret_cast<const uint8_t*>(samples), got);
    }
    // Whole reads only: a partial write would shift every later sample by a byte.
    if (got > 0 && got % 2 == 0 && voice_active && voice_mic_stream != nullptr) {
      if (xStreamBufferSpacesAvailable(voice_mic_stream) >= got) {
        xStreamBufferSend(voice_mic_stream, samples, got, 0);
      } else {
        ++voice_mic_dropped;
      }
    }
  }
}

// ---------------------------------------------------------------- camera (Sense board)

// Size codes match CAMERA_SIZES in src/esp32_link.py. VGA is the largest because the
// JPEG buffers are sized for the frame size given at init.
framesize_t cameraFrameSize(uint8_t code) {
  constexpr framesize_t kSizes[] = {FRAMESIZE_QQVGA, FRAMESIZE_QVGA, FRAMESIZE_HVGA, FRAMESIZE_VGA};
  return code < sizeof(kSizes) / sizeof(kSizes[0]) ? kSizes[code] : FRAMESIZE_QVGA;
}

// The module sits in the head so that the sensor's picture comes out turned 180 deg and
// mirrored left-right. Both together are a vertical flip, done here on the sensor (relative
// to its default, which differs between camera modules), so every consumer of the frames
// (web UI, voice chat, /capture) gets them upright.
void cameraOrient() {
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor == nullptr) return;
  sensor->set_vflip(sensor, sensor->status.vflip ? 0 : 1);
}

// Pins from CAMERA_MODEL_XIAO_ESP32S3 in the core's CameraWebServer example.
bool cameraStart() {
  if (!psramFound()) return false;
  camera_config_t config = {};
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.pin_xclk = 10;
  config.pin_sccb_sda = 40;
  config.pin_sccb_scl = 39;
  config.pin_d7 = 48;
  config.pin_d6 = 11;
  config.pin_d5 = 12;
  config.pin_d4 = 14;
  config.pin_d3 = 16;
  config.pin_d2 = 18;
  config.pin_d1 = 17;
  config.pin_d0 = 15;
  config.pin_vsync = 38;
  config.pin_href = 47;
  config.pin_pclk = 13;
  config.xclk_freq_hz = 20000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&config) != ESP_OK) return false;
  cameraOrient();
  return true;
}

void cameraApply(uint8_t size_code, uint8_t quality) {
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor == nullptr) return;
  sensor->set_framesize(sensor, cameraFrameSize(size_code));
  sensor->set_quality(sensor, quality);
}

void cameraSendFrame(Print* out, const camera_fb_t* fb, uint16_t seq) {
  const uint32_t total = fb->len;
  for (uint32_t offset = 0; offset < total; offset += kCameraChunkBytes) {
    uint8_t head[10];
    memcpy(head, &seq, 2);
    memcpy(head + 2, &offset, 4);
    memcpy(head + 6, &total, 4);
    linkSend(out, protocol::kMsgCameraChunk, head, sizeof(head), fb->buf + offset,
             min<size_t>(kCameraChunkBytes, total - offset));
  }
}

void publishJpeg(const camera_fb_t* fb) {
  if (shared_jpeg == nullptr || fb->len > kSharedJpegBytes) return;
  xSemaphoreTake(jpeg_lock, portMAX_DELAY);
  memcpy(shared_jpeg, fb->buf, fb->len);
  shared_jpeg_size = fb->len;
  shared_jpeg_ms = millis();
  xSemaphoreGive(jpeg_lock);
  ++shared_jpeg_seq;
}

// Status heartbeat always, JPEG frames while requested (micTask also writes to the Pi).
// Runs the camera while the Pi or a Wi-Fi viewer wants it, at the larger of their sizes.
void cameraTask(void*) {
  uint32_t last_status_ms = 0;
  uint32_t last_start_ms = 0;
  uint8_t applied_size = 0xFF;
  uint8_t applied_quality = 0xFF;
  uint16_t seq = 0;
  uint32_t last_pi_frame_ms = 0;
  for (;;) {
    const uint32_t now = millis();
    if (now - last_status_ms >= kStatusIntervalMs) {
      last_status_ms = now;
      sendStatus();
    }
    Print* out = camera_out;
    const bool to_pi = camera_wanted && out != nullptr && now - camera_request_ms < kCameraKeepaliveMs;
    const bool to_wifi = wifi_viewers > 0;
    const bool to_voice = voice_camera_wanted;
    const bool wanted = (to_pi || to_wifi || to_voice) && !ota_active;
    if (!wanted) {
      if (camera_on) {
        esp_camera_deinit();
        camera_on = false;
      }
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }
    if (!camera_on) {
      if (camera_failed && now - last_start_ms < kCameraRetryMs) {
        vTaskDelay(pdMS_TO_TICKS(20));
        continue;
      }
      last_start_ms = now;
      camera_on = cameraStart();
      camera_failed = !camera_on;
      applied_size = applied_quality = 0xFF;
      Serial.printf("camera: %s\n", camera_on ? "streaming" : (psramFound() ? "init failed" : "no PSRAM"));
      if (camera_on) {
        // The first frame after init is often dark while exposure settles.
        camera_fb_t* warm = esp_camera_fb_get();
        if (warm != nullptr) esp_camera_fb_return(warm);
      }
      continue;
    }
    uint8_t size = 0;
    uint8_t quality = 63;
    if (to_pi) {
      size = max(size, uint8_t(camera_size_code));
      quality = min(quality, uint8_t(camera_quality));
    }
    if (to_wifi) {
      size = max(size, uint8_t(wifi_size_code));
      quality = min(quality, uint8_t(wifi_quality));
    }
    if (to_voice) {
      size = max(size, uint8_t(voice_size_code));
      quality = min(quality, uint8_t(voice_quality));
    }
    if (applied_size != size || applied_quality != quality) {
      applied_size = size;
      applied_quality = quality;
      cameraApply(size, quality);
    }
    // A slow Pi viewer (the voice client) leaves the UART free between its frames.
    const bool pi_due = to_pi && now - last_pi_frame_ms >= camera_interval_ms;
    const bool voice_due = to_voice && voice_frame_request;
    if (!pi_due && !to_wifi && !voice_due) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb == nullptr) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    if (to_wifi || voice_due) publishJpeg(fb);
    if (voice_due) voice_frame_request = false;
    if (pi_due) {
      last_pi_frame_ms = now;
      cameraSendFrame(out, fb, seq++);
    }
    esp_camera_fb_return(fb);
  }
}

// ---------------------------------------------------------------- Wi-Fi: camera page + OTA

void wifiConfigSet(const uint8_t* data, uint16_t size, bool persist) {
  xSemaphoreTake(wifi_config_lock, portMAX_DELAY);
  memcpy(wifi_config, data, size);
  wifi_config_size = size;
  xSemaphoreGive(wifi_config_lock);
  wifi_config_crc = protocol::crc32(data, size);
  if (persist) prefs.putBytes("wifi", data, size);
  wifi_config_dirty = true;
}

void wifiConfigLoad() {
  prefs.begin("xiaoxiaobai", false);
  uint8_t data[protocol::kMaxPayload];
  const size_t size = prefs.getBytesLength("wifi");
  if (size == 0 || size > sizeof(data) || prefs.getBytes("wifi", data, size) != size) return;
  if (protocol::wifiConfigValid(data, size)) wifiConfigSet(data, size, false);
}

// Rebuilds the network list from the stored kMsgWifi payload; nullptr when it is empty.
const char* resetReasonName() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON: return "power on";
    case ESP_RST_SW: return "software restart";
    case ESP_RST_PANIC: return "crash (panic)";
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT: return "watchdog";
    case ESP_RST_BROWNOUT: return "BROWNOUT - supply voltage dropped";
    case ESP_RST_USB: return "USB";
    default: return "other";
  }
}

uint8_t wifiNetworksFromConfig(WifiNetwork* out) {
  uint8_t data[protocol::kMaxPayload];
  xSemaphoreTake(wifi_config_lock, portMAX_DELAY);
  const uint16_t size = wifi_config_size;
  memcpy(data, wifi_config, size);
  xSemaphoreGive(wifi_config_lock);
  if (size == 0) return 0;
  size_t pos = 1;
  for (uint8_t i = 0; i < data[0]; ++i) {
    memset(&out[i], 0, sizeof(out[i]));
    memcpy(out[i].ssid, data + pos + 1, data[pos]);
    pos += 1 + data[pos];
    memcpy(out[i].psk, data + pos + 1, data[pos]);
    pos += 1 + data[pos];
  }
  return data[0];
}

// Joins the strongest saved network in range. Reports each step in wifi_state for the Pi,
// since the board is usually not on USB when Wi-Fi needs debugging.
bool wifiJoin(const WifiNetwork* networks, uint8_t count) {
  wifi_state = protocol::kWifiScanning;
  const int16_t found = WiFi.scanNetworks();
  wifi_seen = found > 0 ? uint8_t(found > 255 ? 255 : found) : 0;
  int best = -1;
  int32_t best_rssi = INT32_MIN;
  int32_t channel = 0;
  uint8_t bssid[6] = {};
  for (int16_t i = 0; i < found; ++i) {
    for (uint8_t n = 0; n < count; ++n) {
      if (WiFi.SSID(i) == networks[n].ssid && WiFi.RSSI(i) > best_rssi) {
        best = n;
        best_rssi = WiFi.RSSI(i);
        channel = WiFi.channel(i);
        memcpy(bssid, WiFi.BSSID(i), sizeof(bssid));
      }
    }
  }
  WiFi.scanDelete();
  if (best < 0) {
    wifi_saved_rssi = 0;
    wifi_state = protocol::kWifiNotFound;
    Serial.printf("wifi: scan saw %d networks, none of the %u saved ones\n", found, count);
    return false;
  }
  wifi_saved_rssi = int8_t(best_rssi);
  wifi_reason = 0;
  wifi_state = protocol::kWifiJoining;
  Serial.printf("wifi: joining %s (channel %ld, %ld dBm)\n", networks[best].ssid, long(channel), long(best_rssi));
  WiFi.begin(networks[best].ssid, networks[best].psk[0] ? networks[best].psk : nullptr, channel, bssid);
  const uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < kWifiJoinTimeoutMs && !wifi_config_dirty) {
    vTaskDelay(pdMS_TO_TICKS(50));
  }
  if (WiFi.status() == WL_CONNECTED) {
    wifi_state = protocol::kWifiConnected;
    Serial.printf("wifi: joined %s as %s (http://%s.local/)\n", networks[best].ssid,
                  WiFi.localIP().toString().c_str(), kHostname);
    return true;
  }
  WiFi.disconnect();
  wifi_state = protocol::kWifiJoinFailed;
  Serial.printf("wifi: joining %s failed (reason %u)\n", networks[best].ssid, wifi_reason);
  return false;
}

uint8_t sizeCodeFromName(const char* name, uint8_t fallback) {
  constexpr const char* kNames[] = {"qqvga", "qvga", "hvga", "vga"};
  for (uint8_t i = 0; i < sizeof(kNames) / sizeof(kNames[0]); ++i) {
    if (strcmp(name, kNames[i]) == 0) return i;
  }
  return fallback;
}

// Waits for the camera task's next frame and copies it into `buffer`; 0 on timeout.
size_t waitJpeg(uint32_t& seen, uint8_t* buffer, uint32_t timeout_ms) {
  const uint32_t start = millis();
  while (shared_jpeg_seq == seen) {
    if (millis() - start > timeout_ms) return 0;
    vTaskDelay(pdMS_TO_TICKS(5));
  }
  xSemaphoreTake(jpeg_lock, portMAX_DELAY);
  const size_t size = shared_jpeg_size;
  memcpy(buffer, shared_jpeg, size);
  seen = shared_jpeg_seq;
  xSemaphoreGive(jpeg_lock);
  return size;
}

const char kIndexHtml[] = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>xiaoxiaobai-face camera</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 14px sans-serif; }
  body { display: flex; flex-direction: column; }
  header { display: flex; gap: 8px; align-items: center; padding: 8px; flex-wrap: wrap; }
  button { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  button.on { background: #2a6; border-color: #2a6; }
  #info { margin-left: auto; font-family: monospace; }
  main { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }
  img { max-width: 100%; max-height: 100%; }
</style></head><body>
<header><span>Wi-Fi direct</span><span id="sizes"></span>
<button id="rotate">rotate 90&deg;</button><a href="/capture" download="xiaoxiaobai.jpg"><button>snapshot</button></a>
<span id="info"></span></header>
<main><img id="view" alt="camera"></main>
<script>
const params = new URLSearchParams(location.search);
let size = params.get("size") || "vga";
const quality = params.get("quality") || "12";
let rotation = Number(localStorage.getItem("xiaoxiaobaiCameraRotation") || 0);
const view = document.getElementById("view");
function start() {
  view.src = `//${location.hostname}:81/stream?size=${size}&quality=${quality}&t=${Date.now()}`;
  for (const b of document.querySelectorAll("#sizes button")) b.classList.toggle("on", b.dataset.size === size);
}
for (const s of ["qqvga", "qvga", "hvga", "vga"]) {
  const b = document.createElement("button");
  b.textContent = s.toUpperCase(); b.dataset.size = s;
  b.onclick = () => { size = s; start(); };
  document.getElementById("sizes").appendChild(b);
}
function rotate() { view.style.transform = `rotate(${rotation}deg)`; localStorage.setItem("xiaoxiaobaiCameraRotation", rotation); }
document.getElementById("rotate").onclick = () => { rotation = (rotation + 90) % 360; rotate(); };
view.onerror = () => setTimeout(start, 1000);
async function poll() {
  try {
    const s = await (await fetch("/status")).json();
    document.getElementById("info").textContent = `${s.ssid} ${s.rssi} dBm | camera ${s.camera} | viewers ${s.viewers}`;
  } catch (e) { document.getElementById("info").textContent = "offline"; }
}
rotate(); start(); poll(); setInterval(poll, 2000);
</script></body></html>
)HTML";

esp_err_t indexHandler(httpd_req_t* req) {
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  return httpd_resp_send(req, kIndexHtml, HTTPD_RESP_USE_STRLEN);
}

esp_err_t statusHandler(httpd_req_t* req) {
  char json[192];
  snprintf(json, sizeof(json), "{\"ssid\":\"%s\",\"rssi\":%d,\"camera\":\"%s\",\"viewers\":%d,\"psram\":%s}",
           WiFi.SSID().c_str(), int(WiFi.RSSI()), camera_on ? "on" : (camera_failed ? "failed" : "off"),
           int(wifi_viewers), psramFound() ? "true" : "false");
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

esp_err_t captureHandler(httpd_req_t* req) {
  uint8_t* buffer = static_cast<uint8_t*>(heap_caps_malloc(kSharedJpegBytes, MALLOC_CAP_SPIRAM));
  if (buffer == nullptr) return httpd_resp_send_500(req);
  uint32_t seen = shared_jpeg_seq;
  ++wifi_viewers;
  const size_t size = waitJpeg(seen, buffer, 5000);
  --wifi_viewers;
  esp_err_t result;
  if (size == 0) {
    httpd_resp_set_status(req, "503 Service Unavailable");
    result = httpd_resp_send(req, "camera not ready\n", HTTPD_RESP_USE_STRLEN);
  } else {
    httpd_resp_set_type(req, "image/jpeg");
    result = httpd_resp_send(req, reinterpret_cast<const char*>(buffer), size);
  }
  free(buffer);
  return result;
}

// One viewer at a time: each server runs its handlers on a single task.
esp_err_t streamHandler(httpd_req_t* req) {
  char query[64] = {};
  char value[16] = {};
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    if (httpd_query_key_value(query, "size", value, sizeof(value)) == ESP_OK) {
      wifi_size_code = sizeCodeFromName(value, wifi_size_code);
    }
    if (httpd_query_key_value(query, "quality", value, sizeof(value)) == ESP_OK) {
      wifi_quality = constrain(atoi(value), 6, 63);
    }
  }
  uint8_t* buffer = static_cast<uint8_t*>(heap_caps_malloc(kSharedJpegBytes, MALLOC_CAP_SPIRAM));
  if (buffer == nullptr) return httpd_resp_send_500(req);
  httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  uint32_t seen = shared_jpeg_seq;
  ++wifi_viewers;
  for (;;) {
    const size_t size = waitJpeg(seen, buffer, kStreamIdleMs);
    if (size == 0) break;
    char part[80];
    const int part_size = snprintf(part, sizeof(part),
                                   "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", unsigned(size));
    if (httpd_resp_send_chunk(req, part, part_size) != ESP_OK ||
        httpd_resp_send_chunk(req, reinterpret_cast<const char*>(buffer), size) != ESP_OK ||
        httpd_resp_send_chunk(req, "\r\n", 2) != ESP_OK) {
      break;
    }
  }
  --wifi_viewers;
  free(buffer);
  return ESP_FAIL;  // closes the socket
}

void httpStart() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = kWebPort;
  config.ctrl_port = 32768;
  httpd_handle_t web = nullptr;
  if (httpd_start(&web, &config) == ESP_OK) {
    const httpd_uri_t routes[] = {
        {.uri = "/", .method = HTTP_GET, .handler = indexHandler, .user_ctx = nullptr},
        {.uri = "/status", .method = HTTP_GET, .handler = statusHandler, .user_ctx = nullptr},
        {.uri = "/capture", .method = HTTP_GET, .handler = captureHandler, .user_ctx = nullptr},
    };
    for (const httpd_uri_t& route : routes) httpd_register_uri_handler(web, &route);
  }
  config.server_port = kStreamPort;
  config.ctrl_port = 32769;
  config.lru_purge_enable = true;
  httpd_handle_t stream = nullptr;
  if (httpd_start(&stream, &config) == ESP_OK) {
    const httpd_uri_t route = {.uri = "/stream", .method = HTTP_GET, .handler = streamHandler, .user_ctx = nullptr};
    httpd_register_uri_handler(stream, &route);
  }
}

void otaStart() {
  ArduinoOTA.setHostname(kHostname);
  ArduinoOTA.setPassword(kOtaPassword);
  ArduinoOTA.onStart([]() {
    ota_active = true;
    Serial.println("ota: receiving firmware");
  });
  ArduinoOTA.onEnd([]() { Serial.println("ota: done, rebooting"); });
  ArduinoOTA.onError([](ota_error_t error) {
    ota_active = false;
    Serial.printf("ota: failed (%u)\n", unsigned(error));
  });
  ArduinoOTA.begin();
  MDNS.addService("http", "tcp", kWebPort);
}

// Joins whichever of the Pi's networks is in range, then serves the camera page and OTA.
void wifiTask(void*) {
  static WifiNetwork networks[protocol::kMaxWifiNetworks];
  uint8_t count = 0;
  bool radio_on = false;
  bool services = false;
  uint32_t next_attempt_ms = 0;
  for (;;) {
    if (wifi_config_dirty) {
      wifi_config_dirty = false;
      count = wifiNetworksFromConfig(networks);
      if (count > 0 && !radio_on) {
        WiFi.setHostname(kHostname);
        WiFi.mode(WIFI_STA);
        WiFi.setSleep(false);
        // Reconnects go through wifiJoin(): a driver retry in the background makes its scan fail.
        WiFi.setAutoReconnect(false);
        radio_on = true;
      }
      if (radio_on) WiFi.disconnect();
      if (count == 0) wifi_state = protocol::kWifiNoConfig;
      next_attempt_ms = millis();
      // The USB console usually misses the boot banner, so the reset cause rides along here.
      Serial.printf("wifi: %u network(s) from the Pi (last reset: %s)\n", count, resetReasonName());
    }
    if (wifi_state == protocol::kWifiConnected && WiFi.status() != WL_CONNECTED) {
      wifi_state = protocol::kWifiJoinFailed;
      Serial.printf("wifi: connection lost (reason %u)\n", wifi_reason);
    }
    if (count > 0 && WiFi.status() != WL_CONNECTED && int32_t(millis() - next_attempt_ms) >= 0) {
      if (!wifiJoin(networks, count)) next_attempt_ms = millis() + kWifiRetryMs;
    }
    if (!services && WiFi.status() == WL_CONNECTED) {
      otaStart();
      httpStart();
      services = true;
    }
    if (services && !uart_ota) ArduinoOTA.handle();
    vTaskDelay(pdMS_TO_TICKS(services ? 10 : 100));
  }
}

void wifiInit() {
  wifi_config_lock = xSemaphoreCreateMutex();
  jpeg_lock = xSemaphoreCreateMutex();
  shared_jpeg = static_cast<uint8_t*>(heap_caps_malloc(kSharedJpegBytes, MALLOC_CAP_SPIRAM));
  wifiConfigLoad();
  WiFi.onEvent(
      [](arduino_event_id_t, arduino_event_info_t info) {
        // ASSOC_LEAVE is our own WiFi.disconnect(); keep the reason that made the join fail.
        const uint8_t reason = info.wifi_sta_disconnected.reason;
        if (reason != WIFI_REASON_ASSOC_LEAVE) wifi_reason = reason;
      },
      ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
  xTaskCreatePinnedToCore(wifiTask, "wifi", 8192, nullptr, 1, nullptr, 0);
}

// ---------------------------------------------------------------- direct voice session (Qwen realtime)

SemaphoreHandle_t voice_lock = nullptr;
uint8_t* voice_config_buf = nullptr;
size_t voice_config_received = 0;
std::unique_ptr<VoiceConfig> voice_config_next;  // guarded by voice_lock
std::atomic<uint32_t> voice_config_seq{0};
uint8_t* voice_context_buf = nullptr;
size_t voice_context_received = 0;
char* voice_context_next = nullptr;  // guarded by voice_lock, voice_context_len bytes
size_t voice_context_len = 0;
std::atomic<uint32_t> voice_context_seq{0};
volatile uint32_t voice_keepalive_ms = 0;
volatile bool voice_stopped = true;
Print* volatile voice_out = &link_uart;  // the link the voice client's config came in on

void voiceReport(uint8_t kind, const char* text, size_t size) {
  linkSend(voice_out, protocol::kMsgVoiceEvent, &kind, 1, reinterpret_cast<const uint8_t*>(text), min<size_t>(size, 1024));
}

void voiceReport(uint8_t kind, const std::string& text) { voiceReport(kind, text.data(), text.size()); }

bool parseVoiceConfig(const uint8_t* data, size_t size, VoiceConfig& out) {
  if (size < 8 || data[0] != kVoiceConfigVersion) return false;
  out.tls = data[1] & 1;
  out.full_duplex = data[1] & 2;
  out.camera = data[1] & 4;
  out.port = data[2] | data[3] << 8;
  out.volume = min<uint8_t>(data[4], 100);
  out.mic_gain_x16 = data[5];
  out.camera_size = min<uint8_t>(data[6], 3);
  out.camera_quality = constrain(data[7], 6, 63);
  size_t pos = 8;
  for (std::string* field : {&out.host, &out.omni_path, &out.tts_path, &out.headers, &out.omni_session, &out.tts_session}) {
    if (pos + 2 > size) return false;
    const size_t len = data[pos] | data[pos + 1] << 8;
    pos += 2;
    if (pos + len > size) return false;
    field->assign(reinterpret_cast<const char*>(data + pos), len);
    pos += len;
  }
  return pos == size && !out.host.empty() && !out.omni_path.empty();
}

// Runs on the loop() task, slice by slice; the voice task picks the finished config up.
void voiceConfigChunk(const uint8_t* payload, uint16_t length) {
  if (length < 4 || voice_config_buf == nullptr) return;
  const size_t offset = payload[0] | payload[1] << 8;
  const size_t total = payload[2] | payload[3] << 8;
  const size_t bytes = length - 4;
  if (offset == 0) voice_config_received = 0;
  if (offset != voice_config_received || total > kVoiceMaxConfigBytes || offset + bytes > total) {
    voice_config_received = 0;
    return;
  }
  memcpy(voice_config_buf + offset, payload + 4, bytes);
  voice_config_received += bytes;
  if (voice_config_received != total) return;
  voice_config_received = 0;
  std::unique_ptr<VoiceConfig> config(new VoiceConfig());
  const bool valid = parseVoiceConfig(voice_config_buf, total, *config);
  memset(voice_config_buf, 0, total);  // holds the API key
  if (!valid) {
    voiceReport(protocol::kVoiceEventError, std::string("invalid voice config (firmware and Pi code out of sync?)"));
    return;
  }
  xSemaphoreTake(voice_lock, portMAX_DELAY);
  voice_config_next = std::move(config);
  xSemaphoreGive(voice_lock);
  voice_keepalive_ms = millis();
  voice_stopped = false;
  ++voice_config_seq;
}

// Runs on the loop() task like voiceConfigChunk; voiceSendContext picks the finished event up.
void voiceContextChunk(const uint8_t* payload, uint16_t length) {
  if (length < 4 || voice_context_buf == nullptr) return;
  const size_t offset = payload[0] | payload[1] << 8;
  const size_t total = payload[2] | payload[3] << 8;
  const size_t bytes = length - 4;
  if (offset == 0) voice_context_received = 0;
  if (offset != voice_context_received || total > kVoiceMaxContextBytes || offset + bytes > total) {
    voice_context_received = 0;
    return;
  }
  memcpy(voice_context_buf + offset, payload + 4, bytes);
  voice_context_received += bytes;
  if (voice_context_received != total) return;
  voice_context_received = 0;
  xSemaphoreTake(voice_lock, portMAX_DELAY);
  memcpy(voice_context_next, voice_context_buf, total);
  voice_context_len = total;
  xSemaphoreGive(voice_lock);
  ++voice_context_seq;
}

bool growBuffer(char*& buffer, size_t& cap, size_t need) {
  if (need <= cap) return true;
  const size_t size = max(need, cap * 2);
  void* grown = heap_caps_realloc(buffer, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (grown == nullptr) grown = realloc(buffer, size);
  if (grown == nullptr) return false;
  buffer = static_cast<char*>(grown);
  cap = size;
  return true;
}

void wsClose(WsConn& c) {
  if (c.ws != nullptr) {
    esp_transport_close(c.ws);
    esp_transport_destroy(c.ws);
  }
  if (c.base != nullptr) esp_transport_destroy(c.base);
  c.ws = c.base = nullptr;
  c.len = 0;
}

bool wsOpen(WsConn& c, const VoiceConfig& config, const std::string& path, std::string& error) {
  c.base = config.tls ? esp_transport_ssl_init() : esp_transport_tcp_init();
  if (c.base == nullptr) {
    error = "out of memory";
    return false;
  }
  if (config.tls) esp_transport_ssl_crt_bundle_attach(c.base, esp_crt_bundle_attach);
  c.ws = esp_transport_ws_init(c.base);
  if (c.ws == nullptr) {
    error = "out of memory";
    wsClose(c);
    return false;
  }
  esp_transport_ws_config_t ws_config = {};
  ws_config.ws_path = path.c_str();
  ws_config.headers = config.headers.c_str();
  ws_config.user_agent = "xiaoxiaobai-face";
  esp_transport_ws_set_config(c.ws, &ws_config);
  if (esp_transport_connect(c.ws, config.host.c_str(), config.port, kVoiceConnectTimeoutMs) < 0) {
    const int status = esp_transport_ws_get_upgrade_request_status(c.ws);
    error = status > 0 ? "HTTP " + std::to_string(status) + (status == 401 || status == 403 ? " (API key rejected)" : "")
                       : "cannot reach " + config.host + ":" + std::to_string(config.port);
    wsClose(c);
    return false;
  }
  return true;
}

// esp_transport_ws masks the payload in place (and restores it), so `text` must be writable.
bool wsSend(WsConn& c, char* text, size_t size) {
  return c.ws != nullptr &&
         esp_transport_ws_send_raw(c.ws, ws_transport_opcodes_t(WS_TRANSPORT_OPCODES_TEXT | WS_TRANSPORT_OPCODES_FIN), text,
                                   size, kVoiceIoTimeoutMs) >= 0;
}

// 1: a complete text message is in c.msg (NUL-terminated, c.len bytes), 0: nothing yet, -1: dead.
int wsReceive(WsConn& c) {
  const int ready = esp_transport_poll_read(c.ws, 0);
  if (ready <= 0) return ready;
  if (!growBuffer(c.msg, c.cap, c.len + 4096)) return -1;
  int got = esp_transport_read(c.ws, c.msg + c.len, c.cap - c.len - 1, kVoiceIoTimeoutMs);
  if (got < 0) return -1;
  const int opcode = esp_transport_ws_get_read_opcode(c.ws);
  if (opcode == WS_TRANSPORT_OPCODES_CLOSE) return -1;
  if (opcode >= WS_TRANSPORT_OPCODES_CLOSE) return 0;  // ping / pong, answered by esp_transport_ws
  if (got == 0) return 0;  // header timed out: the opcode and length are the previous frame's
  const size_t payload = esp_transport_ws_get_read_payload_len(c.ws);
  if (c.len + payload > kVoiceMaxMessageBytes || !growBuffer(c.msg, c.cap, c.len + payload + 1)) return -1;
  for (size_t done = got; done < payload; done += got) {
    got = esp_transport_read(c.ws, c.msg + c.len + done, payload - done, kVoiceIoTimeoutMs);
    if (got <= 0) return -1;
  }
  c.len += payload;
  if (!esp_transport_ws_get_fin_flag(c.ws)) return 0;
  c.msg[c.len] = '\0';
  return 1;
}

constexpr const char* kSentenceEnds[] = {"。", "！", "？", "；", "…", "!", "?", ";", "\n"};
constexpr const char* kClauseEnds[] = {"，", "、", "：", ",", ":"};

bool containsAny(const char* text, const char* const* needles, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (strstr(text, needles[i]) != nullptr) return true;
  }
  return false;
}

// Things people say when the answer needs the camera.
constexpr const char* kVisionWords[] = {
    "看", "瞧", "眼前", "面前", "前面", "这是什么", "这是啥", "这个是", "那是什么", "那是啥", "什么东西",
    "颜色", "拍", "照片", "图片", "画面", "认识", "长什么样", "长啥样", "样子", "镜头", "摄像头", "手里",
    "手上", "拿着", "几根", "几个", "多少个", "穿", "识别", "写的什么", "写了什么", "上面写", "念一下", "读一下",
};

bool voiceNeedsLook(const char* text) {
  return text != nullptr && containsAny(text, kVisionWords, std::size(kVisionWords));
}

bool speakerBusy(uint32_t tail_ms) {
  return pcm_stream != nullptr && (xStreamBufferBytesAvailable(pcm_stream) > 0 || millis() - last_pcm_ms < tail_ms);
}

cJSON* voiceEvent(const char* type) {
  cJSON* event = cJSON_CreateObject();
  cJSON_AddStringToObject(event, "type", type);
  return event;
}

// Sends and frees `event`.
bool voiceSend(VoiceSession& s, WsConn& c, cJSON* event) {
  char id[24];
  snprintf(id, sizeof(id), "event_esp%lu", static_cast<unsigned long>(++s.event_seq));
  cJSON_AddStringToObject(event, "event_id", id);
  char* text = cJSON_PrintUnformatted(event);
  cJSON_Delete(event);
  if (text == nullptr) return false;
  const bool sent = wsSend(c, text, strlen(text));
  cJSON_free(text);
  return sent;
}

void voicePlay(VoiceSession& s, const VoiceConfig& config, const char* b64) {
  const size_t b64_len = strlen(b64);
  if (pcm_stream == nullptr || !growBuffer(s.scratch, s.scratch_cap, b64_len / 4 * 3 + 4)) return;
  size_t bytes = 0;
  if (mbedtls_base64_decode(reinterpret_cast<uint8_t*>(s.scratch), s.scratch_cap, &bytes,
                            reinterpret_cast<const uint8_t*>(b64), b64_len) != 0) {
    return;
  }
  bytes &= ~size_t(1);
  int16_t* samples = reinterpret_cast<int16_t*>(s.scratch);
  if (config.volume < 100) {
    for (size_t i = 0; i < bytes / 2; ++i) samples[i] = int16_t(int32_t(samples[i]) * config.volume / 100);
  }
  if (s.turn.latency_pending) {
    s.turn.latency_pending = false;
    const uint32_t now = millis();
    const VoiceTiming& t = s.turn.timing;
    auto since = [&](uint32_t at) { return at ? long(at - t.stopped) : -1L; };
    char line[256];
    snprintf(line, sizeof(line),
             "timing transcript=%ld created=%ld text=%ld commit=%ld tts=%ld audio=%ld images=%lu image_bytes=%lu "
             "image_send_ms=%lu mic_backlog_ms=%lu mic_dropped=%lu",
             since(t.transcript), since(t.created), since(t.text), since(t.commit), since(t.tts_created), since(now),
             static_cast<unsigned long>(t.images), static_cast<unsigned long>(t.image_bytes),
             static_cast<unsigned long>(t.image_send_ms), static_cast<unsigned long>(t.mic_backlog_ms),
             static_cast<unsigned long>(voice_mic_dropped - t.mic_dropped_base));
    voiceReport(protocol::kVoiceEventDebug, std::string(line));
    voiceReport(protocol::kVoiceEventLatency, std::to_string(now - s.turn.speech_stopped_ms));
  }
  xStreamBufferSend(pcm_stream, s.scratch, bytes, pdMS_TO_TICKS(1000));
}

// First time `at` is set during the current reply's wait for audio.
void voiceStamp(const VoiceTurn& turn, uint32_t& at) {
  if (turn.latency_pending && at == 0) at = millis();
}

bool ttsCommit(VoiceSession& s) {
  voiceStamp(s.turn, s.turn.timing.commit);
  TtsState& t = s.tts_state;
  t.uncommitted = false;
  t.clause_bytes = 0;
  t.first_clause = false;
  ++t.outstanding;
  return voiceSend(s, s.tts, voiceEvent("input_text_buffer.commit"));
}

bool ttsSendText(VoiceSession& s, const char* text) {
  TtsState& t = s.tts_state;
  cJSON* event = voiceEvent("input_text_buffer.append");
  cJSON_AddStringToObject(event, "text", text);
  if (!voiceSend(s, s.tts, event)) return false;
  t.uncommitted = true;
  t.clause_bytes += strlen(text);
  if (containsAny(text, kSentenceEnds, std::size(kSentenceEnds)) ||
      (t.first_clause && t.clause_bytes >= kTtsFirstClauseBytes && containsAny(text, kClauseEnds, std::size(kClauseEnds)))) {
    return ttsCommit(s);
  }
  return true;
}

void ttsFeed(VoiceSession& s, const char* text) {
  if (!s.tts_state.ready) {
    s.tts_state.pending += text;
    return;
  }
  ttsSendText(s, text);
}

// End of a reply: synthesize whatever is still buffered.
void ttsFinish(VoiceSession& s) {
  TtsState& t = s.tts_state;
  if (!t.ready) {
    t.finish_pending = !t.pending.empty();
    return;
  }
  if (t.uncommitted) ttsCommit(s);
  t.first_clause = true;
}

void voiceBargeIn(VoiceSession& s) {
  pcm_flush = true;
  s.turn.latency_pending = false;
  if (s.turn.response_active) {
    s.turn.cancelled_id = s.turn.response_id;
    s.turn.response_active = false;
    voiceSend(s, s.omni, voiceEvent("response.cancel"));
  }
  TtsState& t = s.tts_state;
  if (s.tts.ws != nullptr) {
    t.skip = max(0, t.outstanding - (t.in_response ? 1 : 0));
    if (t.in_response) t.stale = true;
    if (t.uncommitted) voiceSend(s, s.tts, voiceEvent("input_text_buffer.clear"));
  }
  t.uncommitted = false;
  t.clause_bytes = 0;
  t.first_clause = true;
  t.pending.clear();
  t.finish_pending = false;
  voiceReport(protocol::kVoiceEventInterrupted, "", 0);
}

// The reply under way is dropped and made again (with the instructions just updated).
void voiceAnswerAgain(VoiceSession& s) {
  if (s.turn.response_active) {
    s.turn.cancelled_id = s.turn.response_id;
    s.turn.response_active = false;
    voiceSend(s, s.omni, voiceEvent("response.cancel"));
  }
  voiceSend(s, s.omni, voiceEvent("response.create"));
}

// Knowledge-base passages for what the user just said (a whole session.update from the Pi): sent
// as is, and the reply made again with them unless it has started.
bool voiceSendContext(VoiceSession& s, uint32_t& seen_seq) {
  if (voice_context_seq == seen_seq) return true;
  xSemaphoreTake(voice_lock, portMAX_DELAY);
  seen_seq = voice_context_seq;
  const size_t len = voice_context_len;
  const bool copied = growBuffer(s.scratch, s.scratch_cap, len + 1);
  if (copied) memcpy(s.scratch, voice_context_next, len);
  xSemaphoreGive(voice_lock);
  if (!copied || len < 2 || !s.turn.omni_ready) return true;
  if (!wsSend(s.omni, s.scratch + 1, len - 1)) return false;
  VoiceTurn& turn = s.turn;
  if (s.scratch[0] != 1 || !turn.latency_pending || turn.timing.text != 0 || speakerBusy(0)) return true;
  if (turn.response_active) {
    voiceAnswerAgain(s);
  } else {
    turn.redo_on_created = true;
  }
  return true;
}

const char* jsonString(const cJSON* object, const char* key) {
  return cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(object, key));
}

void voiceReportError(const cJSON* event, const char* source) {
  char* text = cJSON_PrintUnformatted(cJSON_GetObjectItemCaseSensitive(event, "error"));
  voiceReport(protocol::kVoiceEventError, std::string(source) + ": " + (text ? text : "error"));
  cJSON_free(text);
}

void voiceOmniEvent(VoiceSession& s, const VoiceConfig& config, const cJSON* event) {
  const char* type = jsonString(event, "type");
  if (type == nullptr) return;
  ++s.omni_events;
  const std::string tag = std::string("|") + type + "|";
  if (s.seen_events.find(tag) == std::string::npos && s.seen_events.size() < 2048) {
    s.seen_events += tag;
    voiceReport(protocol::kVoiceEventDebug, std::string("Qwen sent ") + type);
  }
  VoiceTurn& turn = s.turn;
  const bool native = config.tts_path.empty();
  const char* response_id = jsonString(event, "response_id");
  const bool cancelled = response_id != nullptr && turn.cancelled_id == response_id;
  if (strcmp(type, "session.updated") == 0) {
    if (turn.omni_ready) return;  // knowledge-base passages went in
    turn.omni_ready = true;
    voiceReport(protocol::kVoiceEventStatus,
                "ready (free internal heap " + std::to_string(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)) + ")");
  } else if (strcmp(type, "input_audio_buffer.speech_started") == 0) {
    turn.user_speaking = true;
    turn.redo_on_created = false;
    turn.timing = VoiceTiming();
    turn.timing.mic_dropped_base = voice_mic_dropped;
    turn.heard.clear();
    turn.wants_image = turn.image_sent = false;
    voiceReport(protocol::kVoiceEventSpeechStarted, "", 0);
    if (speakerBusy(kVoiceEchoTailMs) || turn.response_active) voiceBargeIn(s);
  } else if (strcmp(type, "input_audio_buffer.speech_stopped") == 0) {
    turn.user_speaking = false;
    turn.speech_stopped_ms = millis();
    turn.latency_pending = true;
    if (turn.wants_image && !turn.image_sent) {
      voiceReport(protocol::kVoiceEventDebug, std::string("vision: the camera frame was not ready before you stopped talking"));
    }
    VoiceTiming& t = turn.timing;
    t.stopped = turn.speech_stopped_ms;
    t.transcript = t.created = t.text = t.commit = t.tts_created = 0;
    voiceReport(protocol::kVoiceEventSpeechStopped, "", 0);
  } else if (strcmp(type, "response.created") == 0) {
    const char* id = jsonString(cJSON_GetObjectItemCaseSensitive(event, "response"), "id");
    turn.response_active = true;
    turn.response_id = id ? id : "";
    voiceStamp(turn, turn.timing.created);
    if (turn.redo_on_created) {
      turn.redo_on_created = false;
      voiceAnswerAgain(s);
    }
  } else if (strcmp(type, "response.audio_transcript.delta") == 0 || strcmp(type, "response.text.delta") == 0) {
    const char* delta = jsonString(event, "delta");
    if (cancelled || delta == nullptr) return;
    voiceStamp(turn, turn.timing.text);
    voiceReport(protocol::kVoiceEventReplyText, delta, strlen(delta));
    if (!native) ttsFeed(s, delta);
  } else if (strcmp(type, "response.audio.delta") == 0) {
    const char* delta = jsonString(event, "delta");
    if (native && !cancelled && delta != nullptr) voicePlay(s, config, delta);
  } else if (strcmp(type, "response.done") == 0) {
    const char* id = jsonString(cJSON_GetObjectItemCaseSensitive(event, "response"), "id");
    if (id != nullptr && turn.cancelled_id == id) {
      turn.cancelled_id.clear();
    } else if (turn.response_active && (id == nullptr || turn.response_id == id)) {
      turn.response_active = false;
      if (!native) ttsFinish(s);
      voiceReport(protocol::kVoiceEventReplyDone, "", 0);
    }
  } else if (strcmp(type, "conversation.item.input_audio_transcription.completed") == 0) {
    const char* transcript = jsonString(event, "transcript");
    voiceStamp(turn, turn.timing.transcript);
    if (config.camera && !turn.wants_image && voiceNeedsLook(transcript)) {
      voiceReport(protocol::kVoiceEventDebug,
                  std::string("vision: heard the request to look only after you stopped talking; no frame this turn"));
    }
    if (transcript != nullptr) voiceReport(protocol::kVoiceEventUserText, transcript, strlen(transcript));
  } else if (strncmp(type, "conversation.item.input_audio_transcription.", 44) == 0 && turn.user_speaking) {
    // Live transcript while the user talks: "delta" pieces, or the whole "text" / "stash" so far.
    const char* delta = jsonString(event, "delta");
    if (delta != nullptr && turn.heard.size() < 1024) turn.heard += delta;
    if (!config.camera || turn.wants_image) return;
    for (const char* text : {turn.heard.c_str(), jsonString(event, "text"), jsonString(event, "stash")}) {
      if (!voiceNeedsLook(text)) continue;
      turn.wants_image = true;
      turn.image_request_seq = shared_jpeg_seq;
      voice_frame_request = true;
      voiceReport(protocol::kVoiceEventDebug, std::string("vision: taking a picture for \"") + text + "\"");
      break;
    }
  } else if (strcmp(type, "error") == 0) {
    voiceReportError(event, "Omni");
  }
}

void voiceTtsEvent(VoiceSession& s, const VoiceConfig& config, const cJSON* event) {
  const char* type = jsonString(event, "type");
  if (type == nullptr) return;
  TtsState& t = s.tts_state;
  if (strcmp(type, "session.updated") == 0) {
    t.ready = true;
    if (!t.pending.empty()) {
      std::string pending;
      pending.swap(t.pending);
      ttsSendText(s, pending.c_str());
    }
    if (t.finish_pending) {
      t.finish_pending = false;
      ttsFinish(s);
    }
  } else if (strcmp(type, "response.created") == 0) {
    voiceStamp(s.turn, s.turn.timing.tts_created);
    t.in_response = true;
    t.stale = t.skip > 0;
    if (t.skip > 0) --t.skip;
  } else if (strcmp(type, "response.audio.delta") == 0) {
    const char* delta = jsonString(event, "delta");
    if (!t.stale && delta != nullptr) voicePlay(s, config, delta);
  } else if (strcmp(type, "response.done") == 0) {
    t.in_response = false;
    t.stale = false;
    if (t.outstanding > 0) --t.outstanding;
  } else if (strcmp(type, "error") == 0) {
    voiceReportError(event, "TTS");
  }
}

// Microphone audio to the Omni session; muted while the speaker plays unless full duplex.
// A backlog goes out in larger messages, and only a few per call, so Qwen's events keep
// being read even when the upload is slow (a loop until the buffer is empty starved them).
bool voiceSendMic(VoiceSession& s, const VoiceConfig& config) {
  static constexpr size_t kHeadBytes = 96;
  static constexpr size_t kBatchBytes = kVoiceMicChunkBytes * kVoiceMicBatchChunks;
  static int16_t samples[kBatchBytes / sizeof(int16_t)];
  static char json[kHeadBytes + 4 * ((kBatchBytes + 2) / 3) + 4];
  for (int message = 0; message < kVoiceMicMessagesPerPump; ++message) {
    const size_t backlog = xStreamBufferBytesAvailable(voice_mic_stream);
    if (backlog < kVoiceMicChunkBytes) break;
    const uint32_t backlog_ms = backlog / (protocol::kMicSampleRate * sizeof(int16_t) / 1000);
    s.mic_backlog_ms = max(s.mic_backlog_ms, backlog_ms);
    if (s.turn.user_speaking || s.turn.latency_pending) s.turn.timing.mic_backlog_ms = max(s.turn.timing.mic_backlog_ms, backlog_ms);
    const size_t bytes = min(backlog / kVoiceMicChunkBytes * kVoiceMicChunkBytes, kBatchBytes);
    xStreamBufferReceive(voice_mic_stream, samples, bytes, 0);
    const uint32_t chunks = bytes / kVoiceMicChunkBytes;
    if (!s.turn.omni_ready) continue;
    if (!config.full_duplex && speakerBusy(kVoiceEchoTailMs)) {
      s.mic_muted += chunks;
      continue;
    }
    const size_t count = bytes / sizeof(int16_t);
    for (size_t i = 0; i < count; ++i) {
      // The PDM microphone sits on a DC offset; follow it (~5 Hz high-pass) and take it out.
      s.mic_dc += ((int32_t(samples[i]) << 8) - s.mic_dc) >> 9;
      const int32_t ac = int32_t(samples[i]) - (s.mic_dc >> 8);
      samples[i] = int16_t(constrain(ac * config.mic_gain_x16 / 16, -32768, 32767));
      const int32_t level = abs(int32_t(samples[i]));
      if (level > s.mic_peak) s.mic_peak = level;
      s.mic_square += uint64_t(int64_t(samples[i]) * samples[i]);
    }
    s.mic_samples += count;
    s.mic_sent += chunks;
    const int head = snprintf(json, kHeadBytes, "{\"event_id\":\"event_mic%lu\",\"type\":\"input_audio_buffer.append\",\"audio\":\"",
                              static_cast<unsigned long>(++s.event_seq));
    size_t encoded = 0;
    mbedtls_base64_encode(reinterpret_cast<uint8_t*>(json + head), sizeof(json) - head, &encoded,
                          reinterpret_cast<const uint8_t*>(samples), bytes);
    memcpy(json + head + encoded, "\"}", 2);
    if (!wsSend(s.omni, json, head + encoded + 2)) return false;
    s.turn.audio_sent = true;
  }
  return true;
}

// One fresh camera frame, once what the user says needs it; it has to go before their speech ends.
bool voiceSendImage(VoiceSession& s, const VoiceConfig& config) {
  VoiceTurn& turn = s.turn;
  const uint32_t now = millis();
  if (!config.camera || !turn.wants_image || turn.image_sent || !turn.user_speaking || !turn.audio_sent ||
      !turn.omni_ready || shared_jpeg_seq == turn.image_request_seq) {
    return true;
  }
  char head[96];
  const int head_size = snprintf(head, sizeof(head), "{\"event_id\":\"event_img%lu\",\"type\":\"input_image_buffer.append\",\"image\":\"",
                                 static_cast<unsigned long>(++s.event_seq));
  xSemaphoreTake(jpeg_lock, portMAX_DELAY);
  const size_t jpeg_size = shared_jpeg_size;
  size_t encoded = 0;
  const bool fits = growBuffer(s.scratch, s.scratch_cap, head_size + 4 * ((jpeg_size + 2) / 3) + 4);
  if (fits) {
    memcpy(s.scratch, head, head_size);
    mbedtls_base64_encode(reinterpret_cast<uint8_t*>(s.scratch + head_size), s.scratch_cap - head_size, &encoded,
                          shared_jpeg, jpeg_size);
  }
  xSemaphoreGive(jpeg_lock);
  if (!fits) return true;
  memcpy(s.scratch + head_size + encoded, "\"}", 2);
  turn.image_sent = true;
  const bool sent = wsSend(s.omni, s.scratch, head_size + encoded + 2);
  ++turn.timing.images;
  turn.timing.image_bytes += jpeg_size;
  turn.timing.image_send_ms += millis() - now;
  return sent;
}

bool voiceOpen(WsConn& c, const VoiceConfig& config, const std::string& path, const std::string& session, const char* name) {
  std::string error;
  if (!wsOpen(c, config, path, error)) {
    voiceReport(protocol::kVoiceEventError,
                std::string(name) + ": " + error + " (free internal heap " +
                    std::to_string(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)) + ")");
    return false;
  }
  std::string update = session;
  if (!wsSend(c, &update[0], update.size())) {
    voiceReport(protocol::kVoiceEventError, std::string(name) + ": session.update failed");
    wsClose(c);
    return false;
  }
  return true;
}

void voiceClose(VoiceSession& s) {
  wsClose(s.omni);
  wsClose(s.tts);
  s.turn = VoiceTurn();
  s.tts_state = TtsState();
  s.seen_events.clear();
  pcm_flush = true;
}

// Handles what both sessions sent, then sends microphone audio and camera frames.
// Returns false when the Omni session died (a dead TTS session is just closed for reconnection).
bool voicePump(VoiceSession& s, const VoiceConfig& config, uint32_t& context_seq) {
  bool worked = false;
  for (WsConn* c : {&s.omni, &s.tts}) {
    for (int i = 0; c->ws != nullptr && i < 8; ++i) {
      const int got = wsReceive(*c);
      if (got < 0) {
        if (c == &s.omni) return false;
        voiceReport(protocol::kVoiceEventError, std::string("TTS connection lost, reconnecting"));
        wsClose(s.tts);
        s.tts_state = TtsState();
        break;
      }
      if (got == 0) break;
      worked = true;
      cJSON* event = cJSON_ParseWithLength(c->msg, c->len);
      c->len = 0;
      if (event == nullptr) continue;
      if (c == &s.omni) {
        voiceOmniEvent(s, config, event);
      } else {
        voiceTtsEvent(s, config, event);
      }
      cJSON_Delete(event);
    }
  }
  if (!voiceSendMic(s, config) || !voiceSendImage(s, config) || !voiceSendContext(s, context_seq)) return false;
  if (millis() - s.stats_ms >= kVoiceStatsIntervalMs) {
    const uint32_t rms = s.mic_samples ? uint32_t(sqrt(double(s.mic_square) / s.mic_samples)) : 0;
    const uint32_t dropped = voice_mic_dropped;
    char line[200];
    snprintf(line, sizeof(line),
             "mic to Qwen: %lu chunks sent, %lu muted (speaker), %lu dropped, backlog up to %lu ms, rms %lu peak %ld of 32767; "
             "%lu events from Qwen",
             static_cast<unsigned long>(s.mic_sent), static_cast<unsigned long>(s.mic_muted),
             static_cast<unsigned long>(dropped - s.mic_dropped_seen), static_cast<unsigned long>(s.mic_backlog_ms),
             static_cast<unsigned long>(rms), static_cast<long>(s.mic_peak), static_cast<unsigned long>(s.omni_events));
    voiceReport(protocol::kVoiceEventDebug, std::string(line));
    s.stats_ms = millis();
    s.mic_dropped_seen = dropped;
    s.mic_backlog_ms = 0;
    s.mic_sent = s.mic_muted = s.mic_samples = s.omni_events = 0;
    s.mic_square = 0;
    s.mic_peak = 0;
  }
  const bool playing = speakerBusy(150);
  if (playing != s.turn.playing) {
    s.turn.playing = playing;
    voiceReport(protocol::kVoiceEventPlaying, playing ? "1" : "0", 1);
  }
  if (!worked) vTaskDelay(pdMS_TO_TICKS(2));
  return true;
}

// Owns both WebSockets: connects while the Pi keeps the session alive, reconnects on errors.
void voiceTask(void*) {
  VoiceSession s;
  std::unique_ptr<VoiceConfig> config;
  uint32_t seen_seq = 0;
  uint32_t context_seq = voice_context_seq;
  uint32_t next_attempt_ms = 0;
  bool wifi_warned = false;
  for (;;) {
    const uint32_t now = millis();
    if (voice_config_seq != seen_seq) {
      xSemaphoreTake(voice_lock, portMAX_DELAY);
      seen_seq = voice_config_seq;
      config = std::move(voice_config_next);
      xSemaphoreGive(voice_lock);
      voiceClose(s);
      next_attempt_ms = now;
      if (config) {
        voice_size_code = config->camera_size;
        voice_quality = config->camera_quality;
      }
    }
    const bool wanted = config && !voice_stopped && now - voice_keepalive_ms < kVoiceKeepaliveMs && !ota_active;
    if (!wanted) {
      if (s.omni.ws != nullptr) {
        voiceClose(s);
        voiceReport(protocol::kVoiceEventStatus, std::string("stopped"));
      }
      voice_active = false;
      voice_camera_wanted = false;
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }
    voice_active = true;
    voice_camera_wanted = config->camera && s.turn.wants_image && !s.turn.image_sent && s.turn.user_speaking;
    if (s.omni.ws == nullptr) {
      uint8_t stale[256];
      while (xStreamBufferReceive(voice_mic_stream, stale, sizeof(stale), 0) > 0) {
      }
      if (int32_t(now - next_attempt_ms) < 0) {
        vTaskDelay(pdMS_TO_TICKS(20));
        continue;
      }
      if (WiFi.status() != WL_CONNECTED) {
        if (!wifi_warned) voiceReport(protocol::kVoiceEventError, std::string("ESP32 is not on Wi-Fi yet"));
        wifi_warned = true;
        next_attempt_ms = now + 1000;
        continue;
      }
      wifi_warned = false;
      voiceReport(protocol::kVoiceEventStatus, "connecting to " + config->host);
      if (!voiceOpen(s.omni, *config, config->omni_path, config->omni_session, "Omni")) {
        next_attempt_ms = millis() + kVoiceRetryMs;
        continue;
      }
    }
    if (!config->tts_path.empty() && s.tts.ws == nullptr &&
        !voiceOpen(s.tts, *config, config->tts_path, config->tts_session, "TTS")) {
      voiceClose(s);
      next_attempt_ms = millis() + kVoiceRetryMs;
      continue;
    }
    if (!voicePump(s, *config, context_seq)) {
      voiceReport(protocol::kVoiceEventError, std::string("Omni connection lost, reconnecting"));
      voiceClose(s);
      next_attempt_ms = millis() + 500;
    }
  }
}

// The core's mbedTLS allocates from internal RAM only; two TLS sessions (Omni + TTS) would take
// what is left of it and the camera's DMA buffers could no longer be allocated.
void* mbedtlsCalloc(size_t count, size_t size) {
  void* block = heap_caps_calloc(count, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  return block != nullptr ? block : heap_caps_calloc(count, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

void voiceInit() {
  if (!psramFound()) return;
  voice_lock = xSemaphoreCreateMutex();
  voice_config_buf = static_cast<uint8_t*>(heap_caps_malloc(kVoiceMaxConfigBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  voice_context_buf = static_cast<uint8_t*>(heap_caps_malloc(kVoiceMaxContextBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  voice_context_next = static_cast<char*>(heap_caps_malloc(kVoiceMaxContextBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  uint8_t* mic_storage =
      static_cast<uint8_t*>(heap_caps_malloc(kVoiceMicBufferBytes + 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  static StaticStreamBuffer_t mic_stream_state;
  if (mic_storage != nullptr) {
    voice_mic_stream =
        xStreamBufferCreateStatic(kVoiceMicBufferBytes, kVoiceMicChunkBytes, mic_storage, &mic_stream_state);
  }
  if (voice_lock == nullptr || voice_config_buf == nullptr || voice_mic_stream == nullptr) return;
  if (voice_context_next == nullptr) voice_context_buf = nullptr;
  // TLS handshakes need a deep stack.
  xTaskCreatePinnedToCore(voiceTask, "voice", 12288, nullptr, 1, nullptr, 1);
}

// ---------------------------------------------------------------- protocol

protocol::Parser usb_parser;
protocol::Parser uart_parser;
#if XIAOXIAOBAI_BLE
protocol::Parser ble_parser;
#endif

// ---------------------------------------------------------------- UART firmware update

// app0 / app1 in the default_8MB partition table.
constexpr uint32_t kOtaPartitionBytes = 0x330000;
// A stalled transfer must not leave the camera and the voice session switched off.
constexpr uint32_t kUartOtaTimeoutMs = 10000;

bool uart_ota_open = false;
uint32_t uart_ota_ms = 0;
uint32_t uart_ota_size = 0;
uint32_t uart_ota_offset = 0;
uint32_t uart_ota_crc = 0xFFFFFFFF;
uint32_t uart_ota_expect = 0;

void otaReply(Print* out, uint8_t state, uint32_t offset, const char* text) {
  uint8_t head[5] = {state, uint8_t(offset), uint8_t(offset >> 8), uint8_t(offset >> 16), uint8_t(offset >> 24)};
  const size_t text_len = text == nullptr ? 0 : strnlen(text, 200);
  linkSend(out, protocol::kMsgOtaStatus, head, sizeof(head), reinterpret_cast<const uint8_t*>(text), text_len);
}

void uartOtaClose(bool abort_update) {
  if (abort_update && uart_ota_open) Update.abort();
  uart_ota_open = false;
  uart_ota = false;
  ota_active = false;
}

uint32_t crc32Update(uint32_t crc, const uint8_t* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return crc;
}

// Streams a firmware image from the Pi over the UART into the inactive OTA slot, then reboots.
void handleOta(const uint8_t* payload, uint16_t length, Print* reply) {
  if (reply == nullptr || length < 1) return;
  uart_ota_ms = millis();
  const uint8_t op = payload[0];
  if (op == protocol::kOtaAbort) {
    uartOtaClose(true);
    otaReply(reply, protocol::kOtaAck, 0, nullptr);
    return;
  }
  if (op == protocol::kOtaBegin) {
    if (length != 9) return;
    if (ota_active && !uart_ota) {
      otaReply(reply, protocol::kOtaError, 0, "Wi-Fi OTA is in progress");
      return;
    }
    uint32_t size = 0;
    uint32_t crc = 0;
    memcpy(&size, payload + 1, 4);
    memcpy(&crc, payload + 5, 4);
    if (size == 0 || size > kOtaPartitionBytes) {
      otaReply(reply, protocol::kOtaError, 0, "firmware image does not fit the OTA slot");
      return;
    }
    uartOtaClose(true);
    esp_task_wdt_reset();
    if (!Update.begin(size)) {
      otaReply(reply, protocol::kOtaError, 0, Update.errorString());
      return;
    }
    uart_ota_open = true;
    uart_ota = true;
    ota_active = true;
    uart_ota_size = size;
    uart_ota_offset = 0;
    uart_ota_crc = 0xFFFFFFFF;
    uart_ota_expect = crc;
    Serial.printf("ota: UART image %u bytes\n", unsigned(size));
    otaReply(reply, protocol::kOtaAck, 0, nullptr);
    return;
  }
  if (!uart_ota_open) {
    otaReply(reply, protocol::kOtaError, 0, "no UART update started");
    return;
  }
  if (op == protocol::kOtaData) {
    if (length < 5) return;
    uint32_t offset = 0;
    memcpy(&offset, payload + 1, 4);
    const uint8_t* data = payload + 5;
    const size_t bytes = length - 5;
    if (offset != uart_ota_offset) {
      // Duplicate or a skipped chunk: the Pi rewinds to the offset we ack.
      otaReply(reply, protocol::kOtaAck, uart_ota_offset, nullptr);
      return;
    }
    if (offset + bytes > uart_ota_size) {
      uartOtaClose(true);
      otaReply(reply, protocol::kOtaError, offset, "chunk past the end of the image");
      return;
    }
    if (bytes > 0 && Update.write(const_cast<uint8_t*>(data), bytes) != bytes) {
      const char* error = Update.errorString();
      uartOtaClose(true);
      otaReply(reply, protocol::kOtaError, offset, error);
      return;
    }
    uart_ota_crc = crc32Update(uart_ota_crc, data, bytes);
    uart_ota_offset += bytes;
    esp_task_wdt_reset();
    otaReply(reply, protocol::kOtaAck, uart_ota_offset, nullptr);
    return;
  }
  if (op == protocol::kOtaEnd) {
    if (uart_ota_offset != uart_ota_size || ~uart_ota_crc != uart_ota_expect) {
      uartOtaClose(true);
      otaReply(reply, protocol::kOtaError, uart_ota_offset, "image incomplete or checksum mismatch");
      return;
    }
    otaReply(reply, protocol::kOtaDone, uart_ota_offset, nullptr);
    link_uart.flush();
    delay(200);
    if (!Update.end(true)) {
      const char* error = Update.errorString();
      uartOtaClose(true);
      otaReply(reply, protocol::kOtaError, uart_ota_offset, error);
      return;
    }
    ESP.restart();
  }
}

void showFace(Print* reply) {
  face_dirty = true;
  face_live = true;
  face_last_ms = millis();
  backlight_wanted = true;
  if (reply != nullptr) face_out = reply;
}

void imageReply(Print* reply, bool shown) {
  if (reply == nullptr) return;
  uint8_t body[5] = {uint8_t(shown ? 1 : 0)};
  memcpy(body + 1, &image_crc, 4);
  linkSend(reply, protocol::kMsgImageStatus, body, sizeof(body), nullptr, 0);
}

void handleImage(const uint8_t* payload, uint16_t length, Print* reply) {
  if (length < 1) return;
  const uint8_t op = payload[0];
  if (op == protocol::kImageHide) {
    if (image_active) face_dirty = true;
    image_active = false;
    return;
  }
  if (op == protocol::kImageBegin) {
    if (length != 13) return;
    memcpy(&image_size, payload + 1, 4);
    memcpy(&image_crc, payload + 5, 4);
    memcpy(&image_show_ms, payload + 9, 4);
    image_received = 0;
    if (image_packed == nullptr) {
      image_packed = static_cast<uint8_t*>(heap_caps_malloc(kImageMaxBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    }
    if (image_frame == nullptr) {
      image_frame = static_cast<uint16_t*>(heap_caps_malloc(kImagePixels * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    }
    if (image_packed == nullptr || image_frame == nullptr || image_size > kImageMaxBytes) image_size = 0;
    return;
  }
  if (op == protocol::kImageData) {
    if (length < 5 || image_size == 0) return;
    uint32_t offset = 0;
    memcpy(&offset, payload + 1, 4);
    const uint32_t count = length - 5;
    // A lost packet leaves a gap: everything after it is refused and kImageEnd fails.
    if (offset != image_received || offset + count > image_size) return;
    memcpy(image_packed + offset, payload + 5, count);
    image_received += count;
    return;
  }
  if (op != protocol::kImageEnd) return;
  const bool ok = image_size > 0 && image_received == image_size &&
                  protocol::crc32(image_packed, image_size) == image_crc &&
                  protocol::unpackPixelRle(image_packed, image_size, image_frame, kImagePixels);
  if (ok) {
    image_active = true;
    image_dirty = true;
    image_until_ms = millis() + min<uint32_t>(max<uint32_t>(image_show_ms, 1), kImageMaxShowMs);
    backlight_wanted = true;
  }
  image_size = 0;
  imageReply(reply, ok);
}

// `reply` is the port the packet came in on (nullptr for Bluetooth, which cannot carry video).
void handleMessage(const protocol::Parser& parser, Print* reply) {
  const uint8_t* payload = parser.payload();
  const uint16_t length = parser.length();
  switch (parser.type()) {
    case protocol::kMsgFace:
      if (length != face::kMaskBytes) return;
      face::maskToLevels(payload, face_levels);
      showFace(reply);
      return;
    case protocol::kMsgFaceRle: {
      uint8_t mask[face::kMaskBytes];
      if (!protocol::unpackZeroRle(payload, length, mask, face::kMaskBytes)) return;
      face::maskToLevels(mask, face_levels);
      showFace(reply);
      return;
    }
    case protocol::kMsgFaceLevels: {
      if (length < 1 || (payload[0] & ~protocol::kFaceCommit) >= face::kBands) return;
      uint8_t* band = face_levels_next + (payload[0] & ~protocol::kFaceCommit) * face::kBandBytes;
      if (!protocol::unpackZeroRle(payload + 1, length - 1, band, face::kBandBytes)) return;
      if (payload[0] & protocol::kFaceCommit) {
        memcpy(face_levels, face_levels_next, sizeof(face_levels));
        showFace(reply);
      }
      return;
    }
    case protocol::kMsgTones:
      if (length < 1 || (length - 1) % 4 != 0 || tone_queue == nullptr) return;
      for (uint16_t i = 1; i < length; i += 4) {
        const Tone tone{uint16_t(payload[i] | payload[i + 1] << 8), uint16_t(payload[i + 2] | payload[i + 3] << 8),
                        payload[0]};
        xQueueSend(tone_queue, &tone, 0);
      }
      return;
    case protocol::kMsgBacklight:
      if (length != 1) return;
      backlight_wanted = payload[0] != 0;
      return;
    case protocol::kMsgBrightness:
      if (length != 1) return;
      brightness_wanted = constrain(payload[0], 1, 100);
      return;
    case protocol::kMsgPcm:
      if (length % 2 != 0 || pcm_stream == nullptr) return;
      xStreamBufferSend(pcm_stream, payload, length, 0);
      return;
    case protocol::kMsgPcmFlush:
      pcm_flush = true;
      return;
    case protocol::kMsgVoiceConfig:
      if (reply != nullptr) voice_out = reply;
      voiceConfigChunk(payload, length);
      return;
    case protocol::kMsgVoiceContext:
      voiceContextChunk(payload, length);
      return;
    case protocol::kMsgVoiceStop:
      voice_stopped = true;
      return;
    case protocol::kMsgVoiceKeepalive:
      if (reply != nullptr) voice_out = reply;
      voice_keepalive_ms = millis();
      if (voice_config_seq == 0 && voice_config_received == 0) voiceReport(protocol::kVoiceEventNeedConfig, "", 0);
      return;
    case protocol::kMsgMic:
      if (length != 1 || reply == nullptr) return;
      mic_out = reply;
      mic_request_ms = millis();
      mic_wanted = payload[0] != 0;
      return;
    case protocol::kMsgCamera:
      if ((length != 3 && length != 4) || reply == nullptr) return;
      camera_size_code = payload[1];
      camera_quality = constrain(payload[2], 6, 63);
      camera_interval_ms = length == 4 ? payload[3] * 10u : 0;
      camera_out = reply;
      camera_request_ms = millis();
      camera_wanted = payload[0] != 0;
      return;
    case protocol::kMsgOta:
      handleOta(payload, length, reply);
      return;
    case protocol::kMsgImage:
      handleImage(payload, length, reply);
      return;
    case protocol::kMsgWifi:
      if (!protocol::wifiConfigValid(payload, length)) return;
      if (wifi_config_size == length && wifi_config_crc == protocol::crc32(payload, length)) return;
      wifiConfigSet(payload, length, true);
      return;
  }
}

void feedParser(protocol::Parser& parser, const uint8_t* data, size_t count, Print* reply) {
  for (size_t i = 0; i < count; ++i) {
    if (parser.feed(data[i])) handleMessage(parser, reply);
  }
}

// ---------------------------------------------------------------- Bluetooth LE

#if XIAOXIAOBAI_BLE
class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) override {
    ble_last_rx_ms = millis();
    xStreamBufferSend(ble_rx, characteristic->getData(), characteristic->getLength(), 0);
  }
};

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server, ble_gap_conn_desc* desc) override {
    ble_conn = desc->conn_handle;
    ble_last_rx_ms = millis();
    // 15-30 ms interval, 4 s supervision timeout.
    server->requestConnParams(desc->conn_handle, 12, 24, 0, 400);
    Serial.println("BLE: connected");
  }

  void onDisconnect(BLEServer*, ble_gap_conn_desc*) override {
    ble_conn = kNoConnection;
    Serial.println("BLE: disconnected, advertising");
  }
};

void bleInit() {
  ble_rx = xStreamBufferCreate(4096, 1);
  BLEDevice::init(kBleName);
  BLEDevice::setMTU(517);
  ble_server = BLEDevice::createServer();
  ble_server->setCallbacks(new ServerCallbacks());
  ble_server->advertiseOnDisconnect(true);
  BLEService* service = ble_server->createService(kBleServiceUuid);
  BLECharacteristic* rx = service->createCharacteristic(
      kBleRxUuid, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new RxCallbacks());
  service->start();
  BLEAdvertising* advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(kBleServiceUuid);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
}

void dropIdleBleClient() {
  const uint16_t conn = ble_conn;
  if (conn == kNoConnection || millis() - ble_last_rx_ms < kBleIdleDisconnectMs) return;
  Serial.println("BLE: client idle, disconnecting");
  ble_last_rx_ms = millis();
  ble_server->disconnect(conn);
}
#endif

// ---------------------------------------------------------------- main

void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(115200);
  link_uart.setRxBufferSize(kLinkRxBufferBytes);
  link_uart.begin(kLinkBaud, SERIAL_8N1, kLinkRx, kLinkTx);
  link_tx_lock = xSemaphoreCreateMutex();
  if (psramFound()) mbedtls_platform_set_calloc_free(mbedtlsCalloc, free);

  constexpr size_t kFrameBytes = size_t(kPanelWidth) * kPanelHeight * 2;
  // Internal RAM is faster, but Wi-Fi, lwIP, the HTTP servers, the camera DMA buffers and the
  // voice session need it, so with PSRAM the framebuffer always goes there.
  constexpr size_t kInternalReserve = 110 * 1024;
  if (!psramFound() && heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >= kFrameBytes + kInternalReserve) {
    framebuffer = static_cast<uint16_t*>(heap_caps_malloc(kFrameBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  }
  const bool framebuffer_internal = framebuffer != nullptr;
  if (framebuffer == nullptr) {
    framebuffer = static_cast<uint16_t*>(heap_caps_malloc(kFrameBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }
  if (framebuffer != nullptr && !framebuffer_internal) {
    back_framebuffer = static_cast<uint16_t*>(heap_caps_malloc(kFrameBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }
  if (!renderer.begin()) framebuffer = nullptr;
  if (framebuffer != nullptr) lcdInit();
  // The SPI push busy-waits ~30 ms per frame, so it runs on the other core than the renderer.
  if (framebuffer != nullptr && back_framebuffer != nullptr) {
    xTaskCreatePinnedToCore(lcdTask, "lcd", 3072, nullptr, 1, &lcd_task, 0);
  }
  micInit();
  audioInit();
#if XIAOXIAOBAI_BLE
  bleInit();
#endif
  wifiInit();
  xTaskCreatePinnedToCore(cameraTask, "camera", 6144, nullptr, 1, nullptr, 0);
  // Above the camera task so a 20 ms chunk never waits behind more than one JPEG slice.
  if (mic_on) xTaskCreatePinnedToCore(micTask, "mic", 4096, nullptr, 2, nullptr, 1);
  voiceInit();
  Serial.printf("xiaoxiaobai-face ready (reset: %s): framebuffer %s, PSRAM %s, mic %s, free internal heap %u\n",
                resetReasonName(),
                framebuffer ? (framebuffer_internal ? "internal" : (lcd_task ? "PSRAM x2" : "PSRAM")) : "MISSING",
                psramFound() ? "ok" : "MISSING (camera off)", mic_on ? "ok" : "FAILED",
                unsigned(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
}

// Internal RAM is what runs out (camera DMA, Wi-Fi, lwIP); PSRAM is plenty.
void logHeap() {
  static uint32_t last_ms = 0;
  if (millis() - last_ms < kHeapLogIntervalMs) return;
  last_ms = millis();
  Serial.printf("heap: internal free %u, largest block %u, lowest %u; PSRAM free %u\n",
                unsigned(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
                unsigned(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)),
                unsigned(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)),
                unsigned(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  if (face_frames > 0) {
    Serial.printf("face: %u frames, render avg %u us, max %u us\n", unsigned(face_frames),
                  unsigned(face_render_us / face_frames), unsigned(face_render_max_us));
    face_frames = face_render_us = face_render_max_us = 0;
  }
}

// The levels to show: the Pi's face, or the loading spinner fading into it (and back out).
const uint8_t* composeFace(uint32_t now) {
  if (face_mix >= 1.0f) return face_levels;
  face::loadingLevels(now / 1000.0f, face_shown);
  const float p = face_mix * face_mix * (3.0f - 2.0f * face_mix);
  for (int i = 0; i < face::kCells; ++i) {
    face_shown[i] = max(uint8_t(lrintf((face_levels[i] & 31) * p)), uint8_t(lrintf(face_shown[i] * (1.0f - p))));
  }
  return face_shown;
}

void showFaceFrame() {
  static bool back_ready = false;
  static uint32_t last_ms = 0;
  static uint32_t loading_frame_ms = 0;
  const uint32_t now = millis();
  if (face_live && now - face_last_ms > kFaceStaleMs) face_live = false;
  const float step = float(now - last_ms) / kFaceFadeMs;
  last_ms = now;
  face_mix = face_live ? min(1.0f, face_mix + step) : max(0.0f, face_mix - step);
  if (face_mix < 1.0f && backlight_wanted && now - loading_frame_ms >= kLoadingFrameMs) {
    loading_frame_ms = now;
    face_dirty = true;
  }
  if (image_active && int32_t(now - image_until_ms) >= 0) {
    image_active = false;
    face_dirty = true;
  }
  // The faces keep coming while a picture is up, so the face is current when it comes back.
  const bool draw = image_active ? image_dirty : face_dirty;
  if (draw && framebuffer != nullptr && !back_ready) {
    uint16_t* target = lcd_task == nullptr ? framebuffer : back_framebuffer;
    if (image_active) {
      image_dirty = false;
      memcpy(target, image_frame, kImagePixels * 2);
    } else {
      face_dirty = false;
      const uint32_t start = micros();
      renderer.render(composeFace(now), target);
      const uint32_t took = micros() - start;
      ++face_frames;
      face_render_us += took;
      face_render_max_us = max(face_render_max_us, took);
    }
    if (lcd_task == nullptr) {
      lcdPushFrame(framebuffer);
      return;
    }
    back_ready = true;
  }
  if (back_ready && !lcd_busy) {
    std::swap(framebuffer, back_framebuffer);
    back_ready = false;
    lcd_busy = true;
    xTaskNotifyGive(lcd_task);
  }
}

void sendSpeakerLevel() {
  static uint32_t sent_seq = 0;
  const uint32_t report = speaker_report.load();
  Print* out = face_out;
  if ((report >> 16) == sent_seq || out == nullptr) return;
  sent_seq = report >> 16;
  const uint8_t body[2] = {uint8_t(report >> 8), uint8_t(report)};
  linkSend(out, protocol::kMsgSpeaker, body, sizeof(body), nullptr, 0);
}

void loop() {
  uint8_t chunk[512];
  bool received = false;
  while (Serial.available() > 0) {
    const size_t count = Serial.read(chunk, min<size_t>(Serial.available(), sizeof(chunk)));
    feedParser(usb_parser, chunk, count, &Serial);
    received = true;
  }
  while (link_uart.available() > 0) {
    const size_t count = link_uart.read(chunk, min<size_t>(link_uart.available(), sizeof(chunk)));
    feedParser(uart_parser, chunk, count, &link_uart);
    received = true;
  }
#if XIAOXIAOBAI_BLE
  size_t count;
  while ((count = xStreamBufferReceive(ble_rx, chunk, sizeof(chunk), 0)) > 0) {
    feedParser(ble_parser, chunk, count, nullptr);
    received = true;
  }
  dropIdleBleClient();
#endif

  showFaceFrame();
  sendSpeakerLevel();
  if (uart_ota && millis() - uart_ota_ms > kUartOtaTimeoutMs) {
    Serial.println("ota: UART update timed out, keeping the current firmware");
    uartOtaClose(true);
  }
  applyBacklight();
  logHeap();
  if (!received) delay(2);
}
