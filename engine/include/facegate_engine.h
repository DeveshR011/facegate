/* FaceGate inference engine - C ABI.
 *
 * One flat C interface so any host (Python/ctypes, C#, Rust, a test harness)
 * can use the engine without sharing a C++ runtime. All functions are thread
 * safe; failures return NULL / a negative code and set fg_last_error().
 *
 * Pipeline (all in this library):
 *   fg_detect   SCRFD-10G      boxes + 5 landmarks (resize/normalise/decode/NMS included)
 *   fg_align    5-point similarity warp to the canonical 112x112 ArcFace crop
 *   fg_embed    ArcFace         512-d unit vectors for the full face and the mask view,
 *                               each averaged with its mirror image
 *   fg_liveness MiniFASNetV2 + V1SE   probability the face is a live person
 */
#ifndef FACEGATE_ENGINE_H
#define FACEGATE_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <wchar.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef FACEGATE_BUILD
#define FG_API __declspec(dllexport)
#else
#define FG_API
#endif

#define FG_ABI_VERSION 1
#define FG_EMBED_DIM 512
#define FG_CROP 112

#define FG_VIEW_FULL 1 /* whole face                                     */
#define FG_VIEW_MASK 2 /* lower face painted out: eyes / brows / forehead */

typedef struct fg_engine fg_engine;

typedef struct fg_face {
    float bbox[4]; /* x1 y1 x2 y2 in source pixels                               */
    float kps[10]; /* left eye, right eye, nose, left mouth, right mouth (x,y)   */
    float score;
} fg_face;

typedef struct fg_config {
    const wchar_t* detector;   /* static-shape SCRFD .onnx (see tools/prepare_models.py) */
    const wchar_t* recognizer; /* static-batch ArcFace .onnx                              */
    const wchar_t* liveness_a; /* MiniFASNetV2 .onnx; NULL disables liveness              */
    const wchar_t* liveness_b; /* MiniFASNetV1SE .onnx                                    */
    int32_t force_cpu;         /* nonzero: never use the GPU                              */
    int32_t cpu_threads;       /* intra-op threads for CPU execution (0 = 4)              */
    float det_score;           /* detection score threshold (0 = 0.5)                     */
    float det_nms;             /* NMS IoU threshold (0 = 0.4)                             */
} fg_config;

typedef struct fg_info {
    char provider[32]; /* "DmlExecutionProvider" or "CPUExecutionProvider" */
    char adapter[128]; /* GPU chosen for DirectML, or "cpu"                */
    int32_t det_w, det_h;
    int32_t rec_batch;
    int32_t has_liveness;
    uint64_t vram_mb;
} fg_info;

/* Where the calling thread's most recent fg_detect / fg_embed / fg_liveness spent its time. */
typedef struct fg_timing {
    float prepare_ms; /* CPU: resize, normalise, tensor packing        */
    float infer_ms;   /* session.Run: GPU work plus host<->device copy */
    float finish_ms;  /* CPU: decode, NMS, softmax, normalisation      */
} fg_timing;

FG_API int32_t fg_abi_version(void);
FG_API const char* fg_last_error(void);
FG_API void fg_last_timing(fg_timing* out);

FG_API fg_engine* fg_create(const fg_config* cfg);
FG_API void fg_destroy(fg_engine* e);
FG_API void fg_get_info(const fg_engine* e, fg_info* out);

/* Detect faces in a BGR8 image. Returns the face count (largest first, at most
 * max_faces; 0 = no limit up to 64) or a negative error code. */
FG_API int32_t fg_detect(fg_engine* e, const uint8_t* bgr, int32_t w, int32_t h, int32_t stride,
                         int32_t max_faces, fg_face* out);

/* Warp the face described by kps[10] into a 112x112 BGR8 crop. */
FG_API int32_t fg_align(const uint8_t* bgr, int32_t w, int32_t h, int32_t stride,
                        const float* kps10, uint8_t* out_112x112x3);

/* Embed an aligned 112x112 BGR8 crop. `views` is a bit mask of FG_VIEW_*; `out`
 * receives FG_EMBED_DIM floats per selected view, ordered full then mask.
 * flip != 0 averages each embedding with that of the mirrored crop. */
FG_API int32_t fg_embed(fg_engine* e, const uint8_t* crop_112x112x3, int32_t views, int32_t flip,
                        float* out);

/* Probability (0..1) that the face inside bbox[4] is live. */
FG_API int32_t fg_liveness(fg_engine* e, const uint8_t* bgr, int32_t w, int32_t h, int32_t stride,
                           const float* bbox4, float* prob);

#ifdef __cplusplus
}
#endif
#endif
