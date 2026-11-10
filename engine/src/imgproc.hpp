// Image operations the pipeline needs, written to match OpenCV's numerics
// (cv2.resize INTER_LINEAR, cv2.warpAffine) so the C++ engine reproduces the
// Python reference to within one grey level. No third-party dependency.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fg {

struct Image {
    const uint8_t* data;  // BGR8, row stride in bytes
    int w, h, stride;
    const uint8_t* at(int x, int y) const { return data + static_cast<size_t>(y) * stride + 3 * x; }
};

inline uint8_t to_u8(float v) { return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, std::floor(v + 0.5f)))); }

// Source taps for one axis of a bilinear resize, with OpenCV's edge handling.
struct Taps {
    std::vector<int> i0, i1;
    std::vector<float> w1;
    Taps(int src_n, int dst_n) : i0(dst_n), i1(dst_n), w1(dst_n) {
        const double scale = static_cast<double>(src_n) / dst_n;
        for (int d = 0; d < dst_n; ++d) {
            const double f = (d + 0.5) * scale - 0.5;
            int s = static_cast<int>(std::floor(f));
            float a = static_cast<float>(f - s);
            if (s < 0) { s = 0; a = 0.0f; }
            if (s >= src_n - 1) { s = src_n - 1; a = 0.0f; }
            i0[d] = s;
            i1[d] = std::min(s + 1, src_n - 1);
            w1[d] = a;
        }
    }
};

// Bilinear sample of one output pixel (3 channels, unrounded).
inline void bilerp_f(const Image& src, const Taps& tx, int y0, int y1, float wy, int dx, float* out) {
    const uint8_t* a = src.at(tx.i0[dx], y0);
    const uint8_t* b = src.at(tx.i1[dx], y0);
    const uint8_t* c = src.at(tx.i0[dx], y1);
    const uint8_t* d = src.at(tx.i1[dx], y1);
    const float wx = tx.w1[dx];
    for (int ch = 0; ch < 3; ++ch) {
        const float top = a[ch] + (b[ch] - a[ch]) * wx;
        const float bot = c[ch] + (d[ch] - c[ch]) * wx;
        out[ch] = top + (bot - top) * wy;
    }
}

// The same, rounded to u8 the way cv2.resize returns it.
inline void bilerp_row(const Image& src, const Taps& tx, int y0, int y1, float wy, int dx, uint8_t* out) {
    float v[3];
    bilerp_f(src, tx, y0, y1, wy, dx, v);
    for (int ch = 0; ch < 3; ++ch) out[ch] = to_u8(v[ch]);
}

// cv2.resize(INTER_LINEAR) of `src` into a dw x dh BGR8 buffer.
inline void resize_bgr(const Image& src, int dw, int dh, uint8_t* dst) {
    const Taps tx(src.w, dw), ty(src.h, dh);
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            bilerp_row(src, tx, ty.i0[y], ty.i1[y], ty.w1[y], x, dst + 3 * (static_cast<size_t>(y) * dw + x));
}

// Resize into the top-left of a (cw x ch) canvas, then BGR->RGB, (v - mean) * scale, planar CHW.
// Everything outside the resized region is the padding pixel 0.
inline void resize_normalise_chw(const Image& src, int nw, int nh, int cw, int ch, float mean, float scale,
                                 float* out) {
    const size_t plane = static_cast<size_t>(cw) * ch;
    std::fill(out, out + 3 * plane, (0.0f - mean) * scale);
    const Taps tx(src.w, nw), ty(src.h, nh);
    float px[3];       // not rounded to u8 first: half a grey level is far below the network's noise floor
    for (int y = 0; y < nh; ++y) {
        const int y0 = ty.i0[y], y1 = ty.i1[y];
        const float wy = ty.w1[y];
        for (int x = 0; x < nw; ++x) {
            bilerp_f(src, tx, y0, y1, wy, x, px);
            const size_t o = static_cast<size_t>(y) * cw + x;
            out[o] = (px[2] - mean) * scale;             // R
            out[plane + o] = (px[1] - mean) * scale;     // G
            out[2 * plane + o] = (px[0] - mean) * scale; // B
        }
    }
}

// Least-squares similarity transform (rotation, uniform scale, translation) mapping
// src[i] -> dst[i]. Closed form of Umeyama's method for 2-D without reflection.
// Output is the forward 2x3 matrix {a, -b, tx, b, a, ty}.
inline void similarity(const float* src, const float* dst, int n, float m[6]) {
    double smx = 0, smy = 0, dmx = 0, dmy = 0;
    for (int i = 0; i < n; ++i) { smx += src[2 * i]; smy += src[2 * i + 1]; dmx += dst[2 * i]; dmy += dst[2 * i + 1]; }
    smx /= n; smy /= n; dmx /= n; dmy /= n;
    double num_a = 0, num_b = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        const double sx = src[2 * i] - smx, sy = src[2 * i + 1] - smy;
        const double dx = dst[2 * i] - dmx, dy = dst[2 * i + 1] - dmy;
        num_a += sx * dx + sy * dy;
        num_b += sx * dy - sy * dx;
        den += sx * sx + sy * sy;
    }
    const double a = den > 1e-12 ? num_a / den : 1.0;
    const double b = den > 1e-12 ? num_b / den : 0.0;
    m[0] = static_cast<float>(a);  m[1] = static_cast<float>(-b);
    m[2] = static_cast<float>(dmx - (a * smx - b * smy));
    m[3] = static_cast<float>(b);  m[4] = static_cast<float>(a);
    m[5] = static_cast<float>(dmy - (b * smx + a * smy));
}

// cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT=0) with forward matrix m.
inline void warp_affine(const Image& src, const float m[6], int dw, int dh, uint8_t* dst) {
    const double det = static_cast<double>(m[0]) * m[4] - static_cast<double>(m[1]) * m[3];
    const double id = det != 0.0 ? 1.0 / det : 0.0;
    const double ia = m[4] * id, ib = -m[1] * id, ic = -m[3] * id, ie = m[0] * id;
    const double itx = -(ia * m[2] + ib * m[5]);
    const double ity = -(ic * m[2] + ie * m[5]);

    auto fetch = [&](int x, int y, int ch) -> float {
        return (x < 0 || y < 0 || x >= src.w || y >= src.h) ? 0.0f : static_cast<float>(src.at(x, y)[ch]);
    };
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const double sx = ia * x + ib * y + itx, sy = ic * x + ie * y + ity;
            const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            const float fx = static_cast<float>(sx - x0), fy = static_cast<float>(sy - y0);
            uint8_t* o = dst + 3 * (static_cast<size_t>(y) * dw + x);
            for (int ch = 0; ch < 3; ++ch) {
                const float top = fetch(x0, y0, ch) * (1 - fx) + fetch(x0 + 1, y0, ch) * fx;
                const float bot = fetch(x0, y0 + 1, ch) * (1 - fx) + fetch(x0 + 1, y0 + 1, ch) * fx;
                o[ch] = to_u8(top * (1 - fy) + bot * fy);
            }
        }
    }
}

}  // namespace fg
