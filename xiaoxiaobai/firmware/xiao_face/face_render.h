// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 AI-FanGe
//
// C++ port of expressions.render_led_rgb(): turns a 64x48 frame of LED levels (one byte per LED,
// colour index << 5 | brightness 0-31) into the LCD's native 240x320 big-endian RGB565 buffer.
// Every LED is a round dot with a hot core and a soft glow; unlit LEDs show as dim grey dots.

#pragma once

// Arduino builds with -Os; the per-pixel loops need -O2.
#pragma GCC optimize("O2")

#include <esp_heap_caps.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace face {

constexpr int kCols = 64;
constexpr int kRows = 48;
constexpr int kCells = kCols * kRows;
constexpr int kMaskBytes = kCells / 8;
// kMsgFaceLevels sends the frame in horizontal bands of kBandRows LED rows (small enough that even
// the worst-case zero-RLE of a band fits in one packet).
constexpr int kBands = 16;
constexpr int kBandRows = kRows / kBands;
constexpr int kBandBytes = kBandRows * kCols;
constexpr int kPitch = 5;
constexpr int kWidth = kCols * kPitch;
constexpr int kHeight = kRows * kPitch;
constexpr uint8_t kLevelMax = 31;

// face_engine.PALETTE
constexpr uint8_t kPalette[8][3] = {
    {255, 236, 210}, {255, 105, 155}, {80, 170, 255}, {255, 72, 48},
    {255, 196, 64},  {120, 230, 110}, {190, 120, 255}, {150, 235, 255},
};
constexpr uint8_t kUnlit[3] = {20, 21, 24};
constexpr uint8_t kHot[3] = {255, 250, 240};
constexpr float kDotRadius = 2.5f;
constexpr float kCoreRadius = 1.0f;
constexpr float kUnlitRadius = 1.45f;
constexpr float kGamma = 1.0f;
constexpr float kHaloGain = 0.8f;
constexpr float kHotGain = 0.4f;

// Legacy 1-bit masks (kMsgFace / kMsgFaceRle): bit (row * kCols + col), LSB first.
inline void maskToLevels(const uint8_t* mask, uint8_t* levels) {
  for (int i = 0; i < kCells; ++i) levels[i] = ((mask[i >> 3] >> (i & 7)) & 1) ? kLevelMax : 0;
}

// Loading spinner (shown until the Pi sends a face): an arc running round a faint ring, stretching
// and shrinking as it goes, brightest at its head. `t` in seconds.
inline void loadingLevels(float t, uint8_t* levels) {
  constexpr float kTwoPi = 6.2831853f;
  constexpr float kCx = kCols / 2.0f, kCy = kRows / 2.0f;
  constexpr float kRadius = 9.5f, kHalfWidth = 1.15f;
  constexpr int kReach = int(kRadius + kHalfWidth) + 2;
  constexpr float kTail = 0.3f, kTrack = 5.0f;
  memset(levels, 0, kCells);
  const float length = 0.7f + 2.8f * (0.5f - 0.5f * cosf(kTwoPi * t / 1.8f));
  const float head = fmodf(t * kTwoPi * 0.55f, kTwoPi) + length / 2.0f;
  const float tail = head - length;
  const float head_x = kRadius * cosf(head), head_y = kRadius * sinf(head);
  const float tail_x = kRadius * cosf(tail), tail_y = kRadius * sinf(tail);
  for (int r = int(kCy) - kReach; r < int(kCy) + kReach; ++r) {
    for (int c = int(kCx) - kReach; c < int(kCx) + kReach; ++c) {
      const float x = c + 0.5f - kCx, y = r + 0.5f - kCy;
      const float off = fabsf(sqrtf(x * x + y * y) - kRadius);
      if (off > kHalfWidth + 1.0f) continue;
      const float ring = fminf(1.0f, fmaxf(0.0f, kHalfWidth + 0.5f - off));
      float along = fmodf(atan2f(y, x) - tail, kTwoPi);
      if (along < 0) along += kTwoPi;
      float light;
      if (along <= length) {
        light = ring * (kTail + (1.0f - kTail) * along / length);
      } else {  // round caps at both ends
        const float to_head = hypotf(x - head_x, y - head_y), to_tail = hypotf(x - tail_x, y - tail_y);
        light = fminf(1.0f, fmaxf(0.0f, kHalfWidth + 0.5f - fminf(to_head, to_tail))) * (to_head < to_tail ? 1.0f : kTail);
      }
      levels[r * kCols + c] = uint8_t(lrintf(fminf(float(kLevelMax), fmaxf(kLevelMax * light, kTrack * ring))));
    }
  }
}

class Renderer {
 public:
  bool begin() {
    for (int value = 0; value < 256; ++value) {
      const int level = value & 31;
      const float b = powf(level / 31.0f, kGamma);
      for (int ch = 0; ch < 3; ++ch) lut_[value][ch] = uint8_t(lrintf(kPalette[value >> 5][ch] * b));
      bright_[value] = uint8_t(lrintf(b * 255.0f));
      hot_[value] = uint16_t(lrintf(b * b * kHotGain * 256.0f));
    }
    for (int sy = 0; sy < kPitch; ++sy) {
      for (int sx = 0; sx < kPitch; ++sx) {
        const float dot = coverage(sx, sy, kDotRadius);
        dot_[sy][sx] = uint16_t(lrintf(dot * 256.0f));
        core_[sy][sx] = uint16_t(lrintf(coverage(sx, sy, kCoreRadius) * 256.0f));
        unlit_[sy][sx] = uint16_t(lrintf(coverage(sx, sy, kUnlitRadius) * 256.0f));
        halo_w_[sy][sx] = uint16_t(lrintf(kHaloGain * (1.0f - 0.6f * dot) * 256.0f));
        const uint8_t grey[3] = {scale(kUnlit[0], unlit_[sy][sx]), scale(kUnlit[1], unlit_[sy][sx]),
                                 scale(kUnlit[2], unlit_[sy][sx])};
        background_[sy][sx] = pack(grey[0], grey[1], grey[2]);
      }
    }
    // Halo sampled bilinearly between LED centres: sub-pixel s of a cell sits at (s - 2) / 5 of a cell.
    for (int s = 0; s < kPitch; ++s) {
      const float pos = (s + 0.5f - kPitch / 2.0f) / kPitch;
      offset_[s] = pos < 0 ? -1 : 0;
      weight_[s] = uint16_t(lrintf((pos - offset_[s]) * 256.0f));
    }
    halo_ = static_cast<uint8_t*>(heap_caps_malloc(kCells * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (halo_ == nullptr) halo_ = static_cast<uint8_t*>(malloc(kCells * 3));
    return halo_ != nullptr;
  }

  void render(const uint8_t* levels, uint16_t* native) {
    blurHalo(levels);
    markActive();
    for (int x = 0; x < kWidth; ++x) {
      const int c = x / kPitch;
      const int sx = x % kPitch;
      // Halo of every LED row, interpolated across to this pixel column.
      const int lo = clampCol(c + offset_[sx]);
      const int hi = clampCol(c + offset_[sx] + 1);
      const uint16_t wx = weight_[sx];
      for (int r = 0; r < kRows; ++r) {
        const uint8_t* a = halo_ + (r * kCols + lo) * 3;
        const uint8_t* b = halo_ + (r * kCols + hi) * 3;
        for (int ch = 0; ch < 3; ++ch) column_[r][ch] = uint8_t((a[ch] * (256 - wx) + b[ch] * wx) >> 8);
      }
      uint16_t* out = native + x * kHeight;
      for (int j = 0; j < kHeight; ++j) {
        const int y = kHeight - 1 - j;
        const int r = y / kPitch;
        const int sy = y % kPitch;
        const int cell = r * kCols + c;
        if (!active_[cell]) {
          out[j] = background_[sy][sx];
          continue;
        }
        const uint8_t value = levels[cell];
        const uint8_t* rgb = lut_[value];
        const int r0 = clampRow(r + offset_[sy]);
        const int r1 = clampRow(r + offset_[sy] + 1);
        const uint16_t wy = weight_[sy];
        const uint16_t dot = dot_[sy][sx];
        const uint16_t hot = uint16_t((core_[sy][sx] * hot_[value]) >> 8);
        const uint16_t grey = uint16_t((unlit_[sy][sx] * (255 - bright_[value])) >> 8);
        const uint16_t halo_w = halo_w_[sy][sx];
        uint8_t px[3];
        for (int ch = 0; ch < 3; ++ch) {
          const uint32_t halo = (column_[r0][ch] * (256 - wy) + column_[r1][ch] * wy) >> 8;
          const uint32_t sum = ((rgb[ch] * dot) >> 8) + ((kHot[ch] * hot) >> 8) + ((halo * halo_w) >> 8) +
                               ((kUnlit[ch] * grey) >> 8);
          px[ch] = sum > 255 ? 255 : uint8_t(sum);
        }
        out[j] = pack(px[0], px[1], px[2]);
      }
    }
  }

 private:
  static float coverage(int sx, int sy, float radius) {
    int inside = 0;
    for (int i = 0; i < 8; ++i) {
      for (int k = 0; k < 8; ++k) {
        const float px = sx + (k + 0.5f) / 8.0f - kPitch / 2.0f;
        const float py = sy + (i + 0.5f) / 8.0f - kPitch / 2.0f;
        inside += px * px + py * py <= radius * radius;
      }
    }
    return inside / 64.0f;
  }

  static uint8_t scale(uint8_t value, uint16_t weight) { return uint8_t((value * weight) >> 8); }
  static int clampCol(int c) { return c < 0 ? 0 : (c >= kCols ? kCols - 1 : c); }
  static int clampRow(int r) { return r < 0 ? 0 : (r >= kRows ? kRows - 1 : r); }

  static uint16_t pack(uint8_t r, uint8_t g, uint8_t b) {
    const uint16_t value = uint16_t((r & 0xF8) << 8) | uint16_t((g & 0xFC) << 3) | uint16_t(b >> 3);
    return uint16_t((value >> 8) | (value << 8));
  }

  // Separable [1 4 6 4 1] / 16 blur of the LED colours, zero outside the panel.
  void blurHalo(const uint8_t* levels) {
    static constexpr uint8_t kKernel[5] = {1, 4, 6, 4, 1};
    for (int r = 0; r < kRows + 2; ++r) {
      if (r < kRows) {
        uint16_t* row = rows_[r % 5];
        for (int c = 0; c < kCols; ++c) {
          uint16_t sum[3] = {0, 0, 0};
          for (int k = 0; k < 5; ++k) {
            const int cc = c + k - 2;
            if (cc < 0 || cc >= kCols) continue;
            const uint8_t* rgb = lut_[levels[r * kCols + cc]];
            for (int ch = 0; ch < 3; ++ch) sum[ch] += kKernel[k] * rgb[ch];
          }
          for (int ch = 0; ch < 3; ++ch) row[c * 3 + ch] = sum[ch];
        }
      }
      const int out_r = r - 2;
      if (out_r < 0) continue;
      for (int c = 0; c < kCols; ++c) {
        for (int ch = 0; ch < 3; ++ch) {
          uint32_t sum = 0;
          for (int k = 0; k < 5; ++k) {
            const int rr = out_r + k - 2;
            if (rr < 0 || rr >= kRows) continue;
            sum += kKernel[k] * rows_[rr % 5][c * 3 + ch];
          }
          halo_[(out_r * kCols + c) * 3 + ch] = uint8_t(sum >> 8);
        }
      }
    }
  }

  // Cells with light anywhere in their 3x3 neighbourhood get the full shading; the rest are plain unlit dots.
  void markActive() {
    uint8_t* glow = glow_;
    for (int i = 0; i < kCells; ++i) glow[i] = (halo_[i * 3] | halo_[i * 3 + 1] | halo_[i * 3 + 2]) != 0;
    for (int r = 0; r < kRows; ++r) {
      for (int c = 0; c < kCols; ++c) {
        uint8_t any = 0;
        for (int dr = -1; dr <= 1 && !any; ++dr) {
          const int rr = r + dr;
          if (rr < 0 || rr >= kRows) continue;
          for (int dc = -1; dc <= 1; ++dc) {
            const int cc = c + dc;
            if (cc >= 0 && cc < kCols && glow[rr * kCols + cc]) {
              any = 1;
              break;
            }
          }
        }
        active_[r * kCols + c] = any;
      }
    }
  }

  uint8_t lut_[256][3];
  uint8_t bright_[256];
  uint16_t hot_[256];
  uint16_t dot_[kPitch][kPitch], core_[kPitch][kPitch], unlit_[kPitch][kPitch], halo_w_[kPitch][kPitch];
  uint16_t background_[kPitch][kPitch];
  int8_t offset_[kPitch];
  uint16_t weight_[kPitch];
  uint8_t* halo_ = nullptr;
  uint16_t rows_[5][kCols * 3];
  uint8_t column_[kRows][3];
  uint8_t active_[kCells];
  uint8_t glow_[kCells];
};

}  // namespace face
