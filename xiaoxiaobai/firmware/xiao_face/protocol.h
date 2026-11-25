// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 AI-FanGe
//
// Pi <-> ESP32 byte stream, shared by the UART, USB CDC and Bluetooth LE (see src/esp32_link.py).
// Packet: A5 5A | type u8 | length u16 LE | payload | XOR of payload bytes.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <initializer_list>

namespace protocol {

constexpr uint8_t kMagic0 = 0xA5;
constexpr uint8_t kMagic1 = 0x5A;
// Pi -> ESP32
constexpr uint8_t kMsgFace = 0x01;       // raw 64x48 mask
constexpr uint8_t kMsgTones = 0x02;      // volume u8, then (freq u16, duration_ms u16) per note
constexpr uint8_t kMsgBacklight = 0x03;  // u8 on/off
constexpr uint8_t kMsgFaceRle = 0x04;    // zero-run-length encoded mask
// enabled u8, size code u8, JPEG quality u8, optional min frame interval u8 (10 ms units); resent as keepalive
constexpr uint8_t kMsgCamera = 0x05;
constexpr uint8_t kMsgWifi = 0x06;      // count u8, then per network: ssid_len u8, ssid, psk_len u8, psk
constexpr uint8_t kMsgPcm = 0x07;       // s16le mono samples at kPcmSampleRate, queued for the speaker
constexpr uint8_t kMsgPcmFlush = 0x08;  // drop the queued speaker samples (no payload)
constexpr uint8_t kMsgMic = 0x09;       // enabled u8; resent as keepalive
// Direct voice session (the ESP32 talks to Qwen itself): offset u16, total u16, then a slice of the
// config (src/voice_chat.py encode_direct_config); the session (re)starts once the last slice is in.
constexpr uint8_t kMsgVoiceConfig = 0x0A;
constexpr uint8_t kMsgVoiceStop = 0x0B;       // no payload
constexpr uint8_t kMsgVoiceKeepalive = 0x0C;  // no payload; the session stops without one for kVoiceKeepaliveMs
// UART firmware update (the Pi streams the .bin; Wi-Fi OTA stays available too).
// op u8: kOtaBegin then size u32 + crc32 u32, kOtaData then offset u32 + bytes, kOtaEnd, kOtaAbort.
constexpr uint8_t kMsgOta = 0x0D;
constexpr uint8_t kOtaBegin = 1;
constexpr uint8_t kOtaData = 2;
constexpr uint8_t kOtaEnd = 3;
constexpr uint8_t kOtaAbort = 4;
// One horizontal band of a 64x48 frame of LED levels (face_render.h): band u8 | 0x80 on the last band
// of a frame (show it), then the zero-RLE of the band's kBandBytes (colour << 5 | brightness per LED).
constexpr uint8_t kMsgFaceLevels = 0x0E;
constexpr uint8_t kFaceCommit = 0x80;
// Knowledge-base passages for what the user just said, sliced like kMsgVoiceConfig (offset u16,
// total u16, bytes) and put together: redo u8 (1: make the reply again with them) followed by a
// whole session.update event for the Omni session.
constexpr uint8_t kMsgVoiceContext = 0x0F;
// Full-screen picture shown instead of the face (memo boards): op u8, then
// kImageBegin: size u32, crc32 u32, show_ms u32; kImageData: offset u32 + bytes; kImageEnd: decode and
// show (answered with kMsgImageStatus); kImageHide: back to the face. The picture is the panel's native
// big-endian RGB565 (as face::Renderer writes it), packed by unpackPixelRle.
constexpr uint8_t kMsgImage = 0x10;
constexpr uint8_t kImageBegin = 1;
constexpr uint8_t kImageData = 2;
constexpr uint8_t kImageEnd = 3;
constexpr uint8_t kImageHide = 4;
constexpr uint8_t kMsgBrightness = 0x11;  // u8 backlight brightness, 1-100 %; resent as keepalive
// ESP32 -> Pi
constexpr uint8_t kMsgCameraChunk = 0x81;  // frame seq u16, offset u32, total u32, JPEG bytes
constexpr uint8_t kMsgMicChunk = 0x83;     // s16le mono samples at kMicSampleRate
constexpr uint8_t kMsgVoiceEvent = 0x84;   // kind u8 (kVoiceEvent*), UTF-8 text
// state u8 (kOtaAck / kOtaError / kOtaDone), next offset u32, optional UTF-8 text
constexpr uint8_t kMsgOtaStatus = 0x85;
constexpr uint8_t kOtaAck = 1;
constexpr uint8_t kOtaError = 2;
constexpr uint8_t kOtaDone = 3;
// What the speaker is playing, ~every 21 ms while PCM plays and once at 0 when it stops (sent to the port
// the face comes from): loudness u8 (20 * log10(rms + 1) * kSpeakerDbScale), zero-crossing rate u8 (x 255).
constexpr uint8_t kMsgSpeaker = 0x86;
constexpr float kSpeakerDbScale = 2.83f;
// Answer to kImageEnd: shown u8 (0: incomplete, checksum or format error), crc32 u32 from kImageBegin.
constexpr uint8_t kMsgImageStatus = 0x87;

constexpr uint8_t kVoiceEventStatus = 1;          // human-readable state
constexpr uint8_t kVoiceEventError = 2;           // human-readable error
constexpr uint8_t kVoiceEventSpeechStarted = 3;
constexpr uint8_t kVoiceEventSpeechStopped = 4;
constexpr uint8_t kVoiceEventReplyText = 5;       // streaming reply text
constexpr uint8_t kVoiceEventReplyDone = 6;
constexpr uint8_t kVoiceEventUserText = 7;        // transcript of what the user said
constexpr uint8_t kVoiceEventPlaying = 8;         // "1" / "0": the speaker started / stopped talking
constexpr uint8_t kVoiceEventLatency = 9;         // ms from end of speech to first reply audio
constexpr uint8_t kVoiceEventInterrupted = 10;
constexpr uint8_t kVoiceEventNeedConfig = 11;     // keepalive without a config (the ESP32 rebooted)
constexpr uint8_t kVoiceEventDebug = 12;          // diagnostics: microphone stats, first sighting of each Omni event
// flags u8 (kStatus*), Wi-Fi IPv4 (4 bytes, 0.0.0.0 offline), RSSI i8, crc32 of the stored kMsgWifi payload u32,
// Wi-Fi state u8 (kWifi*), networks seen in the last scan u8, best saved network RSSI i8 (0 none seen),
// last disconnect reason u8 (wifi_err_reason_t, 0 none), features u8 (kFeature*)
constexpr uint8_t kMsgStatus = 0x82;
constexpr uint8_t kStatusBytes = 15;
constexpr uint8_t kFeatureFaceLevels = 1 << 0;  // kMsgFaceLevels and kMsgSpeaker
constexpr uint8_t kFeatureImage = 1 << 1;       // kMsgImage
constexpr uint8_t kFeatureBrightness = 1 << 2;  // kMsgBrightness
constexpr uint16_t kMaxPayload = 512;
constexpr uint8_t kMaxWifiNetworks = 8;
constexpr uint32_t kPcmSampleRate = 24000;
constexpr uint32_t kMicSampleRate = 16000;

constexpr uint8_t kStatusFramebuffer = 1 << 0;
constexpr uint8_t kStatusPsram = 1 << 1;
constexpr uint8_t kStatusCameraOn = 1 << 2;
constexpr uint8_t kStatusCameraFailed = 1 << 3;
constexpr uint8_t kStatusWifi = 1 << 4;
constexpr uint8_t kStatusOta = 1 << 5;
constexpr uint8_t kStatusMicOn = 1 << 6;
constexpr uint8_t kStatusMicFailed = 1 << 7;

constexpr uint8_t kWifiNoConfig = 0;
constexpr uint8_t kWifiScanning = 1;
constexpr uint8_t kWifiNotFound = 2;  // none of the saved networks in the scan
constexpr uint8_t kWifiJoining = 3;
constexpr uint8_t kWifiJoinFailed = 4;
constexpr uint8_t kWifiConnected = 5;

// Same value as Python's zlib.crc32.
inline uint32_t crc32(const uint8_t* data, size_t size) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

// Checks a kMsgWifi payload: SSIDs 1-32 bytes, passphrases 0-64 bytes, nothing left over.
inline bool wifiConfigValid(const uint8_t* data, uint16_t size) {
  if (size < 1 || data[0] > kMaxWifiNetworks) return false;
  size_t pos = 1;
  for (uint8_t i = 0; i < data[0]; ++i) {
    for (uint8_t max_len : {uint8_t(32), uint8_t(64)}) {
      if (pos >= size) return false;
      const uint8_t len = data[pos++];
      if (len > max_len || (max_len == 32 && len == 0) || pos + len > size) return false;
      pos += len;
    }
  }
  return pos == size;
}

class Parser {
 public:
  // Returns true once a complete packet with a valid checksum has been received.
  bool feed(uint8_t byte) {
    switch (state_) {
      case 0:
        state_ = byte == kMagic0 ? 1 : 0;
        return false;
      case 1:
        state_ = byte == kMagic1 ? 2 : (byte == kMagic0 ? 1 : 0);
        return false;
      case 2:
        type_ = byte;
        state_ = 3;
        return false;
      case 3:
        length_ = byte;
        state_ = 4;
        return false;
      case 4:
        length_ |= uint16_t(byte) << 8;
        position_ = 0;
        checksum_ = 0;
        state_ = length_ > kMaxPayload ? 0 : (length_ ? 5 : 6);
        return false;
      case 5:
        payload_[position_++] = byte;
        checksum_ ^= byte;
        if (position_ == length_) state_ = 6;
        return false;
      default:
        state_ = 0;
        return byte == checksum_;
    }
  }

  uint8_t type() const { return type_; }
  uint16_t length() const { return length_; }
  const uint8_t* payload() const { return payload_; }

 private:
  uint8_t state_ = 0;
  uint8_t type_ = 0;
  uint16_t length_ = 0;
  uint16_t position_ = 0;
  uint8_t checksum_ = 0;
  uint8_t payload_[kMaxPayload];
};

// 0x00 n expands to n zero bytes; any other byte is a literal (esp32_link.zero_rle).
// Writes `out` only when the stream decodes to exactly `out_size` bytes.
inline bool unpackZeroRle(const uint8_t* data, uint16_t size, uint8_t* out, size_t out_size) {
  uint8_t scratch[kMaxPayload];
  if (out_size > sizeof(scratch)) return false;
  size_t written = 0;
  for (uint16_t i = 0; i < size; ++i) {
    if (data[i] != 0) {
      if (written >= out_size) return false;
      scratch[written++] = data[i];
      continue;
    }
    if (++i >= size) return false;
    const uint8_t run = data[i];
    if (run == 0 || written + run > out_size) return false;
    memset(scratch + written, 0, run);
    written += run;
  }
  if (written != out_size) return false;
  memcpy(out, scratch, out_size);
  return true;
}

// kMsgImage pixels (esp32_link.pixel_rle): control byte c, then c & 0x80 ? one pixel repeated
// (c & 0x7F) + 1 times : c + 1 literal pixels; 2 bytes per pixel, copied as they are.
// True only when the stream decodes to exactly `pixels` pixels.
inline bool unpackPixelRle(const uint8_t* data, size_t size, uint16_t* out, size_t pixels) {
  size_t written = 0;
  size_t i = 0;
  while (i < size) {
    const uint8_t control = data[i++];
    const size_t count = (control & 0x7F) + 1u;
    if (written + count > pixels) return false;
    if (control & 0x80) {
      if (i + 2 > size) return false;
      uint16_t pixel;
      memcpy(&pixel, data + i, 2);
      i += 2;
      for (size_t k = 0; k < count; ++k) out[written++] = pixel;
    } else {
      if (i + count * 2 > size) return false;
      memcpy(out + written, data + i, count * 2);
      i += count * 2;
      written += count;
    }
  }
  return written == pixels;
}

}  // namespace protocol
