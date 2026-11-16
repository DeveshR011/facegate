// FaceGate inference engine. See include/facegate_engine.h for the contract.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "facegate_engine.h"
#include "imgproc.hpp"
#include "mask_bitmap.hpp"

using Microsoft::WRL::ComPtr;

namespace {

thread_local std::string g_error;
thread_local fg_timing g_timing{};

void set_error(const std::string& msg) { g_error = msg; }

using Clock = std::chrono::steady_clock;
float ms_since(Clock::time_point t) { return std::chrono::duration<float, std::milli>(Clock::now() - t).count(); }

// Coordinates must be finite and sane before they are cast to int (NaN / huge values are UB there).
bool sane(const float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(v[i]) || std::fabs(v[i]) > 1.0e7f) return false;
    return true;
}

// Copy into a fixed char buffer, always NUL-terminated (the buffer is zeroed first).
void copy_text(char* dst, size_t capacity, const std::string& text) {
    std::memcpy(dst, text.data(), std::min(text.size(), capacity - 1));
}

constexpr int kMaxFaces = 64;
constexpr float kArcfaceDst[10] = {38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f,
                                   71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f};
constexpr int kLivenessSize = 80;
constexpr float kLivenessScaleA = 2.7f;  // MiniFASNetV2
constexpr float kLivenessScaleB = 4.0f;  // MiniFASNetV1SE
constexpr int kStrides[3] = {8, 16, 32};
constexpr int kAnchorsPerCell = 2;

// ---------------------------------------------------------------------------
// GPU selection
// ---------------------------------------------------------------------------

struct GpuInfo {
    std::string name = "unknown GPU";
    uint64_t vram_mb = 0;
};

// The adapter DXGI ranks first for high performance - the one DirectML's
// HighPerformance preference selects. Used for reporting only.
GpuInfo describe_high_performance_gpu() {
    GpuInfo info;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return info;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                   IID_PPV_ARGS(&adapter))))
        return info;
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(adapter->GetDesc1(&desc))) return info;
    char utf8[128] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8) - 1, nullptr, nullptr);
    info.name = utf8;
    info.vram_mb = desc.DedicatedVideoMemory / (1024 * 1024);
    return info;
}

// ---------------------------------------------------------------------------
// One ONNX model with a static input shape
// ---------------------------------------------------------------------------

class Net {
public:
    Net(Ort::Env& env, const wchar_t* path, bool want_gpu, int cpu_threads, const char* what) {
        if (want_gpu) {
            try {
                session_ = create(env, path, true, cpu_threads);
                gpu_ = true;
            } catch (const Ort::Exception&) {
                session_ = Ort::Session{nullptr};  // fall through to CPU
            }
        }
        if (!session_) session_ = create(env, path, false, cpu_threads);

        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_.GetInputNameAllocated(0, alloc).get();
        for (size_t i = 0; i < session_.GetOutputCount(); ++i)
            out_names_.emplace_back(session_.GetOutputNameAllocated(i, alloc).get());
        shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (int64_t d : shape_)
            if (d <= 0)
                throw std::runtime_error(std::string(what) + ": model has a dynamic input shape; "
                                         "run tools/prepare_models.py to fix it");
    }

    bool gpu() const { return gpu_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    size_t input_count() const {
        size_t n = 1;
        for (int64_t d : shape_) n *= static_cast<size_t>(d);
        return n;
    }

    // Runs one inference. `outputs` are valid until the next call on this Net.
    std::vector<Ort::Value> run(float* data) {
        std::lock_guard<std::mutex> lock(mu_);  // DirectML sessions are not re-entrant
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, data, input_count(), shape_.data(), shape_.size());
        std::vector<const char*> out_ptrs;
        for (const auto& n : out_names_) out_ptrs.push_back(n.c_str());
        const char* in_ptr = in_name_.c_str();
        return session_.Run(Ort::RunOptions{nullptr}, &in_ptr, &input, 1, out_ptrs.data(), out_ptrs.size());
    }

private:
    Ort::Session create(Ort::Env& env, const wchar_t* path, bool gpu, int cpu_threads) {
        Ort::SessionOptions so;
        so.SetLogSeverityLevel(3);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Idle worker threads busy-wait by default. On a 16-core hybrid CPU that steals cores from
        // the caller (measured: an unrelated OpenCV call 4x slower, 3-4 ms lost per detect).
        so.AddConfigEntry("session.intra_op.allow_spinning", "0");
        so.AddConfigEntry("session.inter_op.allow_spinning", "0");
        if (gpu) {
            so.SetIntraOpNumThreads(1);   // the CPU only drives the GPU here
            so.SetInterOpNumThreads(1);
            so.DisableMemPattern();
            so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            const OrtDmlApi* dml = nullptr;
            Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                                    reinterpret_cast<const void**>(&dml)));
            OrtDmlDeviceOptions opts{OrtDmlPerformancePreference::HighPerformance, OrtDmlDeviceFilter::Gpu};
            Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML2(so, &opts));
        } else {
            so.SetIntraOpNumThreads(cpu_threads > 0 ? cpu_threads : 4);
            so.SetInterOpNumThreads(1);
        }
        return Ort::Session(env, path, so);
    }

    Ort::Session session_{nullptr};
    std::string in_name_;
    std::vector<std::string> out_names_;
    std::vector<int64_t> shape_;
    bool gpu_ = false;
    std::mutex mu_;
};

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

struct Candidate {
    float x1, y1, x2, y2, score;
    float kps[10];
};

std::vector<int> nms(const std::vector<Candidate>& c, float thresh) {
    std::vector<int> order(c.size());
    for (size_t i = 0; i < c.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return c[a].score > c[b].score; });
    std::vector<int> keep;
    std::vector<char> dead(c.size(), 0);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int i = order[oi];
        if (dead[i]) continue;
        keep.push_back(i);
        const float ai = (c[i].x2 - c[i].x1 + 1) * (c[i].y2 - c[i].y1 + 1);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const int j = order[oj];
            if (dead[j]) continue;
            const float xx1 = std::max(c[i].x1, c[j].x1), yy1 = std::max(c[i].y1, c[j].y1);
            const float xx2 = std::min(c[i].x2, c[j].x2), yy2 = std::min(c[i].y2, c[j].y2);
            const float w = std::max(0.0f, xx2 - xx1 + 1), h = std::max(0.0f, yy2 - yy1 + 1);
            const float inter = w * h;
            const float aj = (c[j].x2 - c[j].x1 + 1) * (c[j].y2 - c[j].y1 + 1);
            if (inter / (ai + aj - inter) > thresh) dead[j] = 1;
        }
    }
    return keep;
}

}  // namespace

struct fg_engine {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "facegate"};
    std::unique_ptr<Net> det, rec, live_a, live_b;
    float det_score = 0.5f, det_nms = 0.4f;
    int det_w = 0, det_h = 0, rec_batch = 0;
    GpuInfo gpu;
    std::string provider;

    std::array<std::vector<float>, 3> anchors_x, anchors_y;  // per stride, anchor centres (read-only after create)

    void build_anchors() {
        for (int s = 0; s < 3; ++s) {
            const int gw = det_w / kStrides[s], gh = det_h / kStrides[s];
            auto& ax = anchors_x[s];
            auto& ay = anchors_y[s];
            ax.clear(); ay.clear();
            for (int y = 0; y < gh; ++y)
                for (int x = 0; x < gw; ++x)
                    for (int a = 0; a < kAnchorsPerCell; ++a) {
                        ax.push_back(static_cast<float>(x * kStrides[s]));
                        ay.push_back(static_cast<float>(y * kStrides[s]));
                    }
        }
    }
};

namespace {

int detect_impl(fg_engine* e, const fg::Image& img, int max_faces, fg_face* out) {
    const int W = e->det_w, H = e->det_h;
    const float scale = std::min(static_cast<float>(W) / img.w, static_cast<float>(H) / img.h);
    const int nw = std::max(1, static_cast<int>(std::lround(img.w * scale)));
    const int nh = std::max(1, static_cast<int>(std::lround(img.h * scale)));

    const auto t0 = Clock::now();
    thread_local std::vector<float> blob;      // per thread: callers must not share the input buffer
    blob.resize(static_cast<size_t>(3) * W * H);
    fg::resize_normalise_chw(img, std::min(nw, W), std::min(nh, H), W, H, 127.5f, 1.0f / 128.0f,
                             blob.data());
    g_timing.prepare_ms = ms_since(t0);

    const auto t1 = Clock::now();
    std::vector<Ort::Value> outs = e->det->run(blob.data());
    g_timing.infer_ms = ms_since(t1);
    if (outs.size() < 9) throw std::runtime_error("detector returned an unexpected number of outputs");
    // The decode below indexes these tensors by anchor; never trust a model file's output sizes.
    for (int s = 0; s < 3; ++s) {
        const size_t n = e->anchors_x[s].size();
        if (outs[s].GetTensorTypeAndShapeInfo().GetElementCount() < n ||
            outs[s + 3].GetTensorTypeAndShapeInfo().GetElementCount() < 4 * n ||
            outs[s + 6].GetTensorTypeAndShapeInfo().GetElementCount() < 10 * n)
            throw std::runtime_error("detector output does not match its input size");
    }
    const auto t2 = Clock::now();

    std::vector<Candidate> cands;
    for (int s = 0; s < 3; ++s) {
        const float* scores = outs[s].GetTensorData<float>();
        const float* boxes = outs[s + 3].GetTensorData<float>();
        const float* kps = outs[s + 6].GetTensorData<float>();
        const size_t n = e->anchors_x[s].size();
        const float stride = static_cast<float>(kStrides[s]);
        for (size_t i = 0; i < n; ++i) {
            if (scores[i] < e->det_score) continue;
            const float cx = e->anchors_x[s][i], cy = e->anchors_y[s][i];
            Candidate c;
            c.score = scores[i];
            c.x1 = (cx - boxes[4 * i] * stride) / scale;
            c.y1 = (cy - boxes[4 * i + 1] * stride) / scale;
            c.x2 = (cx + boxes[4 * i + 2] * stride) / scale;
            c.y2 = (cy + boxes[4 * i + 3] * stride) / scale;
            for (int k = 0; k < 5; ++k) {
                c.kps[2 * k] = (cx + kps[10 * i + 2 * k] * stride) / scale;
                c.kps[2 * k + 1] = (cy + kps[10 * i + 2 * k + 1] * stride) / scale;
            }
            cands.push_back(c);
        }
    }
    std::vector<int> keep = nms(cands, e->det_nms);  // already sorted by score
    std::sort(keep.begin(), keep.end(), [&](int a, int b) {
        const float aa = (cands[a].x2 - cands[a].x1) * (cands[a].y2 - cands[a].y1);
        const float ab = (cands[b].x2 - cands[b].x1) * (cands[b].y2 - cands[b].y1);
        return aa > ab;  // largest face first: the person nearest the camera
    });

    const int limit = max_faces > 0 ? std::min(max_faces, kMaxFaces) : kMaxFaces;
    const int count = std::min<int>(static_cast<int>(keep.size()), limit);
    for (int i = 0; i < count; ++i) {
        const Candidate& c = cands[keep[i]];
        out[i].bbox[0] = std::min(std::max(c.x1, 0.0f), static_cast<float>(img.w - 1));
        out[i].bbox[1] = std::min(std::max(c.y1, 0.0f), static_cast<float>(img.h - 1));
        out[i].bbox[2] = std::min(std::max(c.x2, 0.0f), static_cast<float>(img.w - 1));
        out[i].bbox[3] = std::min(std::max(c.y2, 0.0f), static_cast<float>(img.h - 1));
        std::memcpy(out[i].kps, c.kps, sizeof(c.kps));
        out[i].score = c.score;
    }
    g_timing.finish_ms = ms_since(t2);
    return count;
}

// Crop around the face the way MiniFASNet was trained (context scale 2.7 / 4.0), resize to 80x80.
void liveness_crop(const fg::Image& img, const float* bbox, float ctx_scale, uint8_t* out80) {
    const float bw = std::max(1.0f, bbox[2] - bbox[0]), bh = std::max(1.0f, bbox[3] - bbox[1]);
    const float s = std::min({static_cast<float>(img.h - 1) / bh, static_cast<float>(img.w - 1) / bw, ctx_scale});
    const float nw = bw * s, nh = bh * s;
    const float cx = bbox[0] + bw / 2, cy = bbox[1] + bh / 2;
    const int x1 = std::max(0, static_cast<int>(cx - nw / 2));
    const int y1 = std::max(0, static_cast<int>(cy - nh / 2));
    const int x2 = std::min(img.w - 1, static_cast<int>(cx + nw / 2));
    const int y2 = std::min(img.h - 1, static_cast<int>(cy + nh / 2));
    if (x2 < x1 || y2 < y1) throw std::runtime_error("face box lies outside the image");
    const fg::Image region{img.at(x1, y1), x2 - x1 + 1, y2 - y1 + 1, img.stride};
    fg::resize_bgr(region, kLivenessSize, kLivenessSize, out80);
}

float live_probability(Net& net, const fg::Image& img, const float* bbox, float ctx_scale) {
    thread_local std::vector<uint8_t> crop;
    thread_local std::vector<float> blob;
    crop.resize(3 * kLivenessSize * kLivenessSize);
    blob.resize(crop.size());
    liveness_crop(img, bbox, ctx_scale, crop.data());
    const size_t plane = static_cast<size_t>(kLivenessSize) * kLivenessSize;
    for (size_t i = 0; i < plane; ++i)          // BGR HWC u8 -> BGR CHW float, no normalisation
        for (int c = 0; c < 3; ++c) blob[c * plane + i] = static_cast<float>(crop[3 * i + c]);
    std::vector<Ort::Value> outs = net.run(blob.data());
    // MiniFASNet has three classes (two kinds of spoof around "real" at index 1); the
    // softmax must run over all of them or a spoof class would be ignored entirely.
    const size_t classes = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (classes < 2) throw std::runtime_error("liveness model returned too few classes");
    const float* logits = outs[0].GetTensorData<float>();
    const float peak = *std::max_element(logits, logits + classes);
    float sum = 0.0f, real = 0.0f;
    for (size_t i = 0; i < classes; ++i) {
        const float e = std::exp(logits[i] - peak);
        sum += e;
        if (i == 1) real = e;
    }
    return real / sum;
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

#define FG_TRY try {
#define FG_CATCH(fail_value)                                                     \
    }                                                                            \
    catch (const Ort::Exception& ex) { set_error(std::string("onnxruntime: ") + ex.what()); return fail_value; } \
    catch (const std::exception& ex) { set_error(ex.what()); return fail_value; } \
    catch (...) { set_error("unknown error"); return fail_value; }

extern "C" {

int32_t fg_abi_version(void) { return FG_ABI_VERSION; }

const char* fg_last_error(void) { return g_error.c_str(); }

void fg_last_timing(fg_timing* out) {
    if (out) *out = g_timing;
}

fg_engine* fg_create(const fg_config* cfg) {
    FG_TRY
    if (!cfg || !cfg->detector || !cfg->recognizer) { set_error("detector and recognizer paths are required"); return nullptr; }
    auto e = std::make_unique<fg_engine>();
    e->det_score = cfg->det_score > 0 ? cfg->det_score : 0.5f;
    e->det_nms = cfg->det_nms > 0 ? cfg->det_nms : 0.4f;
    const bool want_gpu = cfg->force_cpu == 0;
    if (want_gpu) e->gpu = describe_high_performance_gpu();

    // One session at a time, and below one warm-up at a time. Building them on four threads saved
    // about half a second, but it put the GPU driver's device setup on several threads at once:
    // after a power-source change a load hung and the process died with heap corruption (0xC0000374).
    auto make = [&](const wchar_t* path, const char* what) {
        return std::make_unique<Net>(e->env, path, want_gpu, cfg->cpu_threads, what);
    };
    e->det = make(cfg->detector, "detector");
    e->rec = make(cfg->recognizer, "recognizer");
    if (cfg->liveness_a && cfg->liveness_b) {
        e->live_a = make(cfg->liveness_a, "liveness A");
        e->live_b = make(cfg->liveness_b, "liveness B");
    }

    const auto& ds = e->det->shape();
    if (ds.size() != 4) throw std::runtime_error("detector input must be NCHW");
    e->det_h = static_cast<int>(ds[2]);
    e->det_w = static_cast<int>(ds[3]);
    if (e->det_w % 32 || e->det_h % 32) throw std::runtime_error("detector size must be a multiple of 32");
    e->build_anchors();

    const auto& rs = e->rec->shape();
    if (rs.size() != 4 || rs[1] != 3 || rs[2] != FG_CROP || rs[3] != FG_CROP)
        throw std::runtime_error("recognizer input must be Nx3x112x112");
    e->rec_batch = static_cast<int>(rs[0]);

    e->provider = e->rec->gpu() ? "DmlExecutionProvider" : "CPUExecutionProvider";
    if (!e->rec->gpu()) { e->gpu.name = "cpu"; e->gpu.vram_mb = 0; }

    // The first run of a DirectML graph compiles it; pay that now, not during an unlock.
    fg_engine* raw = e.get();
    std::vector<uint8_t> warm_img(static_cast<size_t>(3) * 480 * 640, 0);
    fg_face faces[kMaxFaces];
    detect_impl(raw, fg::Image{warm_img.data(), 640, 480, 640 * 3}, 1, faces);
    std::vector<uint8_t> crop(3 * FG_CROP * FG_CROP, 0);
    std::vector<float> out(2 * FG_EMBED_DIM);
    if (fg_embed(raw, crop.data(), FG_VIEW_FULL | FG_VIEW_MASK, 1, out.data()) < 0)
        throw std::runtime_error(g_error);
    if (raw->live_a) {
        const float box[4] = {200, 100, 400, 350};
        float p;
        if (fg_liveness(raw, warm_img.data(), 640, 480, 640 * 3, box, &p) < 0)
            throw std::runtime_error(g_error);
    }
    return e.release();
    FG_CATCH(nullptr)
}

void fg_destroy(fg_engine* e) { delete e; }

void fg_get_info(const fg_engine* e, fg_info* out) {
    if (!e || !out) return;
    std::memset(out, 0, sizeof(*out));
    copy_text(out->provider, sizeof(out->provider), e->provider);
    copy_text(out->adapter, sizeof(out->adapter), e->gpu.name);
    out->det_w = e->det_w;
    out->det_h = e->det_h;
    out->rec_batch = e->rec_batch;
    out->has_liveness = e->live_a ? 1 : 0;
    out->vram_mb = e->gpu.vram_mb;
}

int32_t fg_detect(fg_engine* e, const uint8_t* bgr, int32_t w, int32_t h, int32_t stride, int32_t max_faces,
                  fg_face* out) {
    FG_TRY
    if (!e || !bgr || !out || w < 8 || h < 8 || stride < 3 * w) { set_error("invalid arguments"); return -1; }
    return detect_impl(e, fg::Image{bgr, w, h, stride}, max_faces, out);
    FG_CATCH(-2)
}

int32_t fg_align(const uint8_t* bgr, int32_t w, int32_t h, int32_t stride, const float* kps10,
                 uint8_t* out) {
    FG_TRY
    if (!bgr || !kps10 || !out || w < 1 || h < 1 || stride < 3 * w || !sane(kps10, 10)) {
        set_error("invalid arguments");
        return -1;
    }
    float m[6];
    fg::similarity(kps10, kArcfaceDst, 5, m);
    fg::warp_affine(fg::Image{bgr, w, h, stride}, m, FG_CROP, FG_CROP, out);
    return 0;
    FG_CATCH(-2)
}

int32_t fg_embed(fg_engine* e, const uint8_t* crop, int32_t views, int32_t flip, float* out) {
    FG_TRY
    if (!e || !crop || !out || (views & ~(FG_VIEW_FULL | FG_VIEW_MASK)) || !views) {
        set_error("invalid arguments");
        return -1;
    }
    const auto t0 = Clock::now();
    constexpr int pixels = FG_CROP * FG_CROP;
    const int view_count = ((views & FG_VIEW_FULL) ? 1 : 0) + ((views & FG_VIEW_MASK) ? 1 : 0);
    const int per_view = flip ? 2 : 1;
    const int total = view_count * per_view;
    const int batch = e->rec_batch;

    // Build every (view, mirror) crop as a normalised RGB CHW tensor.
    std::vector<float> all(static_cast<size_t>(total) * 3 * pixels);
    int slot = 0;
    for (int v = 0; v < 2; ++v) {
        if (!(views & (1 << v))) continue;
        for (int mirror = 0; mirror < per_view; ++mirror, ++slot) {
            float* dst = all.data() + static_cast<size_t>(slot) * 3 * pixels;
            for (int y = 0; y < FG_CROP; ++y) {
                const char* mrow = fg::kMaskRows[y];
                for (int x = 0; x < FG_CROP; ++x) {
                    const int sx = mirror ? FG_CROP - 1 - x : x;
                    const uint8_t* p = crop + 3 * (static_cast<size_t>(y) * FG_CROP + sx);
                    const bool painted = (v == 1) && mrow[sx] == '#';
                    const size_t o = static_cast<size_t>(y) * FG_CROP + x;
                    for (int c = 0; c < 3; ++c) {
                        const float px = painted ? 127.0f : static_cast<float>(p[2 - c]);   // BGR -> RGB
                        dst[c * pixels + o] = (px - 127.5f) / 127.5f;
                    }
                }
            }
        }
    }

    // Run in chunks of the model's fixed batch, padding the last chunk.
    std::vector<float> feats(static_cast<size_t>(total) * FG_EMBED_DIM);
    std::vector<float> chunk(static_cast<size_t>(batch) * 3 * pixels);
    g_timing.prepare_ms = ms_since(t0);
    const auto t1 = Clock::now();
    for (int start = 0; start < total; start += batch) {
        const int n = std::min(batch, total - start);
        std::memcpy(chunk.data(), all.data() + static_cast<size_t>(start) * 3 * pixels,
                    sizeof(float) * n * 3 * pixels);
        for (int pad = n; pad < batch; ++pad)
            std::memcpy(chunk.data() + static_cast<size_t>(pad) * 3 * pixels,
                        chunk.data() + static_cast<size_t>(n - 1) * 3 * pixels, sizeof(float) * 3 * pixels);
        std::vector<Ort::Value> outs = e->rec->run(chunk.data());
        std::memcpy(feats.data() + static_cast<size_t>(start) * FG_EMBED_DIM, outs[0].GetTensorData<float>(),
                    sizeof(float) * n * FG_EMBED_DIM);
    }

    g_timing.infer_ms = ms_since(t1);
    const auto t2 = Clock::now();

    // Sum each view with its mirror, then L2-normalise.
    for (int v = 0; v < view_count; ++v) {
        float* dst = out + static_cast<size_t>(v) * FG_EMBED_DIM;
        double norm = 0;
        for (int d = 0; d < FG_EMBED_DIM; ++d) {
            float s = feats[static_cast<size_t>(v * per_view) * FG_EMBED_DIM + d];
            if (flip) s += feats[static_cast<size_t>(v * per_view + 1) * FG_EMBED_DIM + d];
            dst[d] = s;
            norm += static_cast<double>(s) * s;
        }
        const float inv = static_cast<float>(1.0 / (std::sqrt(norm) + 1e-10));
        for (int d = 0; d < FG_EMBED_DIM; ++d) dst[d] *= inv;
    }
    g_timing.finish_ms = ms_since(t2);
    return 0;
    FG_CATCH(-2)
}

int32_t fg_liveness(fg_engine* e, const uint8_t* bgr, int32_t w, int32_t h, int32_t stride, const float* bbox,
                    float* prob) {
    FG_TRY
    if (!e || !bgr || !bbox || !prob || w < 8 || h < 8 || stride < 3 * w || !sane(bbox, 4)) {
        set_error("invalid arguments");
        return -1;
    }
    if (!e->live_a) { set_error("liveness models were not loaded"); return -3; }
    const fg::Image img{bgr, w, h, stride};
    *prob = 0.5f * (live_probability(*e->live_a, img, bbox, kLivenessScaleA) +
                    live_probability(*e->live_b, img, bbox, kLivenessScaleB));
    return 0;
    FG_CATCH(-2)
}

}  // extern "C"
