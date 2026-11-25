// State of the direct voice session (ESP32 <-> Qwen realtime), see the voice section of xiao_face.ino.
// In a header because the Arduino builder puts its function prototypes above the .ino's own types.
#pragma once

#include <esp_transport.h>

#include <cstddef>
#include <cstdint>
#include <string>

// Config from the Pi (src/voice_chat.py encode_direct_config): version u8, flags u8 (bit 0 TLS,
// bit 1 full duplex, bit 2 camera), port u16, volume u8, microphone gain x16 u8, camera size
// code u8, JPEG quality u8, then u16-length-prefixed strings: host, Omni path, TTS path (empty
// for an Omni voice), extra HTTP headers (Authorization), Omni session.update, TTS session.update.
constexpr uint8_t kVoiceConfigVersion = 1;

struct VoiceConfig {
  bool tls = true;
  bool full_duplex = false;
  bool camera = true;
  uint16_t port = 443;
  uint8_t volume = 100;
  uint8_t mic_gain_x16 = 32;
  uint8_t camera_size = 3;
  uint8_t camera_quality = 14;
  std::string host, omni_path, tts_path, headers, omni_session, tts_session;
};

// One WebSocket (esp_transport_ws over TLS) plus the buffer its current message is assembled in.
struct WsConn {
  esp_transport_handle_t base = nullptr;
  esp_transport_handle_t ws = nullptr;
  char* msg = nullptr;
  size_t len = 0;
  size_t cap = 0;
};

struct TtsState {
  bool ready = false;
  std::string pending;  // reply text that arrived before the session was ready
  bool finish_pending = false;
  bool uncommitted = false;
  size_t clause_bytes = 0;
  bool first_clause = true;
  int outstanding = 0;  // commits whose response is not done yet (commit mode: one response each)
  int skip = 0;         // responses still to come that belong to an interrupted reply
  bool in_response = false;
  bool stale = false;
};

// When each step of a reply happened (millis(), 0 not yet), reported with the first reply audio.
struct VoiceTiming {
  uint32_t stopped = 0;      // server VAD: the user's speech ended
  uint32_t transcript = 0;   // transcript of the user's speech
  uint32_t created = 0;      // Omni started the reply
  uint32_t text = 0;         // first reply text
  uint32_t commit = 0;       // first clause sent to the TTS
  uint32_t tts_created = 0;  // TTS started synthesizing it
  uint32_t images = 0;       // camera frames sent while the user talked
  uint32_t image_bytes = 0;
  uint32_t image_send_ms = 0;
  uint32_t mic_backlog_ms = 0;   // most microphone audio waiting on the ESP32 (upload falling behind)
  uint32_t mic_dropped_base = 0;  // voice_mic_dropped when the user started talking
};

struct VoiceTurn {
  bool omni_ready = false;
  bool user_speaking = false;
  bool audio_sent = false;
  bool playing = false;
  bool response_active = false;
  bool latency_pending = false;
  bool redo_on_created = false;  // knowledge-base passages came in before the reply was created
  std::string response_id;
  std::string cancelled_id;
  uint32_t speech_stopped_ms = 0;
  uint32_t camera_until_ms = 0;    // unused; the camera runs only while one requested frame is in flight
  std::string heard;               // live transcript of the user's current speech
  bool wants_image = false;        // what the user says needs a look: send one frame this turn
  bool image_sent = false;
  uint32_t image_request_seq = 0;  // shared_jpeg_seq when the frame was asked for
  VoiceTiming timing;
};

struct VoiceSession {
  WsConn omni;
  WsConn tts;
  VoiceTurn turn;
  TtsState tts_state;
  char* scratch = nullptr;
  size_t scratch_cap = 0;
  uint32_t event_seq = 0;
  int32_t mic_dc = 0;  // microphone DC offset, x256
  // Diagnostics, reported every kVoiceStatsIntervalMs.
  uint32_t stats_ms = 0;
  uint32_t mic_sent = 0;
  uint32_t mic_muted = 0;
  uint32_t mic_samples = 0;
  uint64_t mic_square = 0;
  int32_t mic_peak = 0;
  uint32_t mic_backlog_ms = 0;
  uint32_t mic_dropped_seen = 0;
  uint32_t omni_events = 0;
  std::string seen_events;  // "|type|" for every Omni event type already reported
};
