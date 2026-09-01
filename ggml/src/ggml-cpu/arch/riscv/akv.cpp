#include "akv.h"

#include "../../ggml-cpu-impl.h"
#include "../../vec.h"
#include "akv/akv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

enum class support_reason {
    supported,
    runtime_disabled,
    capability,
    threading,
    feature,
    shape,
    layout,
    mask,
};

struct trace_counters {
    std::atomic<uint64_t>                candidates{};
    std::atomic<uint64_t>                executed{};
    std::atomic<uint64_t>                groups{};
    std::atomic<uint64_t>                executed_v1{};
    std::atomic<uint64_t>                executed_v2{};
    std::atomic<uint64_t>                groups_v1{};
    std::atomic<uint64_t>                groups_v2{};
    std::atomic<uint64_t>                kv_group_tokens{};
    std::atomic<uint64_t>                attention_macs{};
    std::array<std::atomic<uint64_t>, 7> fallback{};
};

enum class kernel_policy {
    automatic,
    v1,
    v2,
    invalid,
};

static trace_counters trace;

static bool env_true(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "yes") == 0 || std::strcmp(value, "YES") == 0 || std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0;
}

static bool emulation_enabled() {
    return env_true("GGML_RISCV_AKV_EMULATE");
}

static bool runtime_enabled() {
    return env_true("GGML_RISCV_AKV") || emulation_enabled();
}

static kernel_policy requested_kernel_policy() {
    const char * value = std::getenv("GGML_RISCV_AKV_KERNEL");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
        return kernel_policy::automatic;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "v1") == 0 || std::strcmp(value, "V1") == 0) {
        return kernel_policy::v1;
    }
    if (std::strcmp(value, "2") == 0 || std::strcmp(value, "v2") == 0 || std::strcmp(value, "V2") == 0) {
        return kernel_policy::v2;
    }
    return kernel_policy::invalid;
}

static void report_trace() {
    if (!env_true("GGML_RISCV_AKV_TRACE")) {
        return;
    }
    std::fprintf(stderr,
                 "GGML_RISCV_AKV_COVERAGE candidate_ops=%llu executed_ops=%llu groups=%llu "
                 "executed_v1=%llu executed_v2=%llu groups_v1=%llu groups_v2=%llu "
                 "kv_group_tokens=%llu attention_macs=%llu "
                 "fallback_runtime=%llu fallback_capability=%llu fallback_threading=%llu "
                 "fallback_feature=%llu fallback_shape=%llu fallback_layout=%llu fallback_mask=%llu\n",
                 static_cast<unsigned long long>(trace.candidates.load()),
                 static_cast<unsigned long long>(trace.executed.load()),
                 static_cast<unsigned long long>(trace.groups.load()),
                 static_cast<unsigned long long>(trace.executed_v1.load()),
                 static_cast<unsigned long long>(trace.executed_v2.load()),
                 static_cast<unsigned long long>(trace.groups_v1.load()),
                 static_cast<unsigned long long>(trace.groups_v2.load()),
                 static_cast<unsigned long long>(trace.kv_group_tokens.load()),
                 static_cast<unsigned long long>(trace.attention_macs.load()),
                 static_cast<unsigned long long>(trace.fallback[0].load()),
                 static_cast<unsigned long long>(trace.fallback[1].load()),
                 static_cast<unsigned long long>(trace.fallback[2].load()),
                 static_cast<unsigned long long>(trace.fallback[3].load()),
                 static_cast<unsigned long long>(trace.fallback[4].load()),
                 static_cast<unsigned long long>(trace.fallback[5].load()),
                 static_cast<unsigned long long>(trace.fallback[6].load()));
}

static void register_trace() {
    static const bool registered = [] {
        std::atexit(report_trace);
        return true;
    }();
    (void) registered;
}

static size_t fallback_index(support_reason reason) {
    return static_cast<size_t>(reason) - static_cast<size_t>(support_reason::runtime_disabled);
}

static akv_device_t detect_device() {
    akv_device_t       device{};
    const akv_status_t status =
        emulation_enabled() ? akv_device_init_reference(&device) : akv_device_query(akv_native_info, nullptr, &device);
    return status == AKV_STATUS_OK ? device : akv_device_t{};
}

static const akv_device_t & device() {
    static const akv_device_t value = detect_device();
    return value;
}

static int active_prefix(const ggml_tensor * mask, int64_t kv_length) {
    const auto * values = static_cast<const ggml_fp16_t *>(mask->data);
    int64_t      active = 0;
    while (active < kv_length && values[active] != UINT16_C(0xfc00)) {
        if ((values[active] & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) {
            return -1;
        }
        ++active;
    }
    for (int64_t index = active; index < kv_length; ++index) {
        if (values[index] != UINT16_C(0xfc00)) {
            return -1;
        }
    }
    return active > 0 && active <= UINT16_MAX ? static_cast<int>(active) : -1;
}

static support_reason supports(const ggml_compute_params * params,
                               const ggml_tensor *        dst,
                               int *                      active_kv,
                               uint32_t *                 kernel_version,
                               uint32_t *                 q_rows,
                               uint32_t *                 head_dim) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!runtime_enabled()) {
        return support_reason::runtime_disabled;
    }
    const akv_capabilities_t & caps = device().capabilities;
    if (!caps.valid || !caps.enabled || !caps.f16_payload || caps.context_count == 0) {
        return support_reason::capability;
    }
    switch (requested_kernel_policy()) {
        case kernel_policy::automatic:
            *kernel_version = caps.token_axis_valid && caps.token_axis_enabled && caps.token_axis_tail &&
                                      caps.token_axis_row_view
                                  ? AKV_ATTENTION_KERNEL_VERSION_V2
                                  : AKV_ATTENTION_KERNEL_VERSION_V1;
            break;
        case kernel_policy::v1:
            *kernel_version = AKV_ATTENTION_KERNEL_VERSION_V1;
            break;
        case kernel_policy::v2:
            if (!caps.token_axis_valid || !caps.token_axis_enabled || !caps.token_axis_tail ||
                !caps.token_axis_row_view) {
                return support_reason::capability;
            }
            *kernel_version = AKV_ATTENTION_KERNEL_VERSION_V2;
            break;
        case kernel_policy::invalid:
            return support_reason::feature;
    }
    if (params->nth <= 0 || params->ith < 0 || params->ith >= params->nth) {
        return support_reason::threading;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    float scale         = 1.0f;
    std::memcpy(&scale, reinterpret_cast<const float *>(dst->op_params), sizeof(float));
    std::memcpy(&max_bias, reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(float));
    std::memcpy(&logit_softcap, reinterpret_cast<const float *>(dst->op_params) + 2, sizeof(float));
    if (params->use_ref || sinks != nullptr || !std::isfinite(scale) || scale <= 0.0f || max_bias != 0.0f ||
        logit_softcap != 0.0f) {
        return support_reason::feature;
    }
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr || q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32) {
        return support_reason::shape;
    }
    if ((q->ne[0] != AKV_HEAD_DIM_64 && q->ne[0] != AKV_HEAD_DIM_96 &&
         q->ne[0] != AKV_HEAD_DIM_128) ||
        q->ne[1] != 1 || q->ne[2] <= 0 ||
        q->ne[3] != 1 || k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0] || k->ne[1] <= 0 ||
        k->ne[1] != v->ne[1] || k->ne[2] <= 0 || k->ne[2] != v->ne[2] || k->ne[3] != 1 || v->ne[3] != 1 ||
        q->ne[2] % k->ne[2] != 0 || dst->ne[0] != q->ne[0] || dst->ne[1] != q->ne[2] || dst->ne[2] != 1 ||
        dst->ne[3] != 1) {
        return support_reason::shape;
    }
    const int64_t shape_q_rows = q->ne[2] / k->ne[2];
    if (shape_q_rows <= 0 || shape_q_rows > UINT32_MAX) {
        return support_reason::shape;
    }
    *q_rows  = static_cast<uint32_t>(shape_q_rows);
    *head_dim = static_cast<uint32_t>(q->ne[0]);
    if (*kernel_version == AKV_ATTENTION_KERNEL_VERSION_V1) {
        if (*q_rows != AKV_ATTENTION_KERNEL_Q_ROWS || *head_dim != AKV_HEAD_DIM_128) {
            return support_reason::shape;
        }
    } else if (!akv_attention_v2_shape_supported(*q_rows, *head_dim)) {
        return support_reason::shape;
    }
    const bool head_dim_capable =
        *head_dim == AKV_HEAD_DIM_64
            ? caps.head_dim_64
            : (*head_dim == AKV_HEAD_DIM_96
                   ? caps.token_axis_d_axis_tail
                   : caps.head_dim_128);
    if (!head_dim_capable || caps.max_q_rows < *q_rows) {
        return support_reason::capability;
    }
    if (q->data == nullptr || k->data == nullptr || v->data == nullptr || mask->data == nullptr ||
        dst->data == nullptr || q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) ||
        v->nb[0] != sizeof(ggml_fp16_t) || mask->nb[0] != sizeof(ggml_fp16_t) || dst->nb[0] != sizeof(float) ||
        q->nb[2] < static_cast<size_t>(*head_dim) * sizeof(float) ||
        k->nb[1] < static_cast<size_t>(*head_dim) * sizeof(ggml_fp16_t) ||
        v->nb[1] < static_cast<size_t>(*head_dim) * sizeof(ggml_fp16_t) || k->nb[1] > UINT32_MAX ||
        v->nb[1] > UINT32_MAX || dst->nb[1] < static_cast<size_t>(*head_dim) * sizeof(float) ||
        dst->nb[1] > UINT32_MAX ||
        (reinterpret_cast<uintptr_t>(q->data) | q->nb[2]) % alignof(float) != 0 ||
        (reinterpret_cast<uintptr_t>(k->data) | reinterpret_cast<uintptr_t>(v->data) |
         reinterpret_cast<uintptr_t>(mask->data) | k->nb[1] | v->nb[1] | k->nb[2] | v->nb[2]) %
                alignof(ggml_fp16_t) !=
            0 ||
        (reinterpret_cast<uintptr_t>(dst->data) | dst->nb[1]) % alignof(float) != 0) {
        return support_reason::layout;
    }
    // The v2 context writes one 128-bit fill beat into a 256-bit bank row.
    // Query rows are copied into the aligned local buffer below; K/V bases and
    // both strides must already satisfy the stronger descriptor contract.  Do
    // this before any worker returns true so an unsupported layout takes the
    // ordinary GGML path on every worker instead of failing during planning.
    const uintptr_t v2_layout_bits = reinterpret_cast<uintptr_t>(k->data) |
                                     reinterpret_cast<uintptr_t>(v->data) | k->nb[1] | v->nb[1] | k->nb[2] | v->nb[2];
    if (*kernel_version == AKV_ATTENTION_KERNEL_VERSION_V2 && (v2_layout_bits & UINT64_C(31)) != 0) {
        return support_reason::layout;
    }
    if (mask->ne[0] < k->ne[1] || mask->ne[1] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return support_reason::mask;
    }
    *active_kv = active_prefix(mask, k->ne[1]);
    return *active_kv > 0 ? support_reason::supported : support_reason::mask;
}

static akv_status_t emulate(void *,
                            const akv_descriptor_t * descriptor,
                            const uint16_t *         mask,
                            float *                  output,
                            size_t                   output_stride,
                            float                    scale) {
    for (uint32_t head = 0; head < descriptor->q_rows; ++head) {
        const auto * query = reinterpret_cast<const ggml_fp16_t *>(static_cast<uintptr_t>(descriptor->q_base) +
                                                                   head * descriptor->q_row_stride_bytes);
        std::array<ggml_fp16_t, AKV_HEAD_DIM_128> accumulator{};
        float                                     maximum = -INFINITY;
        float                                     sum     = 0.0f;

        for (uint32_t token = 0; token < descriptor->kv_length; ++token) {
            const auto * key   = reinterpret_cast<const ggml_fp16_t *>(static_cast<uintptr_t>(descriptor->k_base) +
                                                                       token * descriptor->k_token_stride_bytes);
            const auto * value = reinterpret_cast<const ggml_fp16_t *>(static_cast<uintptr_t>(descriptor->v_base) +
                                                                       token * descriptor->v_token_stride_bytes);
            float        score = 0.0f;
            ggml_vec_dot_f16(descriptor->head_dim, &score, 0, const_cast<ggml_fp16_t *>(key), 0,
                             const_cast<ggml_fp16_t *>(query), 0, 1);
            score = score * scale + GGML_CPU_FP16_TO_FP32(mask[token]);

            if (token == 0) {
                maximum = score;
                sum     = 1.0f;
                std::copy(value, value + descriptor->head_dim, accumulator.begin());
                continue;
            }

            const float new_maximum = std::max(maximum, score);
            const float old_factor  = std::exp(maximum - new_maximum);
            const float new_factor  = std::exp(score - new_maximum);
            ggml_vec_scale_f16(descriptor->head_dim, accumulator.data(), old_factor);
            ggml_vec_mad_f16(descriptor->head_dim, accumulator.data(), value, new_factor);
            sum     = sum * old_factor + new_factor;
            maximum = new_maximum;
        }

        auto *      row     = reinterpret_cast<float *>(reinterpret_cast<char *>(output) + head * output_stride);
        const float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
        for (uint32_t element = 0; element < descriptor->head_dim; ++element) {
            row[element] = GGML_CPU_FP16_TO_FP32(accumulator[element]) * inverse;
        }
    }
    return AKV_STATUS_OK;
}

static akv_status_t execute(const akv_attention_plan_t * plan, akv_attention_v2_workspace_t * workspace) {
    if (plan->kernel_version == AKV_ATTENTION_KERNEL_VERSION_V2) {
        return emulation_enabled() ? akv_attention_execute_v2(plan, emulate, nullptr)
                                   : akv_attention_execute_v2_native(plan, workspace);
    }
    return emulation_enabled() ? akv_attention_execute(plan, emulate, nullptr) : akv_attention_execute_native(plan);
}

}  // namespace

bool ggml_riscv_akv_flash_attn(const ggml_compute_params * params, ggml_tensor * dst) {
    register_trace();
    const bool trace_enabled = env_true("GGML_RISCV_AKV_TRACE");
    if (trace_enabled && params->ith == 0) {
        trace.candidates.fetch_add(1, std::memory_order_relaxed);
    }

    int                  active_kv      = 0;
    uint32_t             kernel_version = 0;
    uint32_t             q_rows         = 0;
    uint32_t             head_dim       = 0;
    const support_reason reason = supports(params, dst, &active_kv, &kernel_version, &q_rows, &head_dim);
    if (reason != support_reason::supported) {
        if (trace_enabled && params->ith == 0) {
            trace.fallback[fallback_index(reason)].fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    // AKV owns one hidden context.
    // Worker zero executes while other workers wait at GGML's post-node barrier.
    if (params->ith != 0) {
        return true;
    }

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    float               scale = 1.0f;
    std::memcpy(&scale, dst->op_params, sizeof(float));

    alignas(64) std::array<ggml_fp16_t, AKV_MAX_Q_ROWS * AKV_HEAD_DIM_128> query{};
    alignas(AKV_DESCRIPTOR_BYTES) akv_attention_plan_t                                  plan{};
    alignas(AKV_DESCRIPTOR_BYTES) akv_attention_v2_workspace_t                          workspace{};
    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(GGML_TYPE_F16)->from_float;

    for (int64_t kv_head = 0; kv_head < k->ne[2]; ++kv_head) {
        const int64_t first_q_head = kv_head * q_rows;
        for (uint32_t head = 0; head < q_rows; ++head) {
            const auto * source =
                reinterpret_cast<const float *>(static_cast<const char *>(q->data) + (first_q_head + head) * q->nb[2]);
            from_float(source, query.data() + head * head_dim, head_dim);
        }

        const akv_attention_problem_t problem = {
            reinterpret_cast<const uint16_t *>(query.data()),
            reinterpret_cast<const uint16_t *>(static_cast<const char *>(k->data) + kv_head * k->nb[2]),
            reinterpret_cast<const uint16_t *>(static_cast<const char *>(v->data) + kv_head * v->nb[2]),
            static_cast<const uint16_t *>(mask->data),
            reinterpret_cast<float *>(static_cast<char *>(dst->data) + first_q_head * dst->nb[1]),
            head_dim * sizeof(ggml_fp16_t),
            k->nb[1],
            v->nb[1],
            dst->nb[1],
            q_rows,
            head_dim,
            static_cast<uint32_t>(active_kv),
            scale,
        };
        const akv_status_t plan_status =
            kernel_version == AKV_ATTENTION_KERNEL_VERSION_V2
                ? akv_attention_plan_create_v2(&device(), &problem, &plan)
                : akv_attention_plan_create(&device(), &problem, &plan);
        if (plan_status != AKV_STATUS_OK) {
            GGML_ABORT("AKV plan creation failed: %s", akv_status_string(plan_status));
        }
        const akv_status_t execute_status = execute(&plan, &workspace);
        if (execute_status != AKV_STATUS_OK) {
            GGML_ABORT("AKV execution failed: %s", akv_status_string(execute_status));
        }
    }

    if (trace_enabled) {
        const uint64_t groups         = static_cast<uint64_t>(k->ne[2]);
        const uint64_t kv_group_tokens = groups * static_cast<uint64_t>(active_kv);
        const uint64_t attention_macs = UINT64_C(2) * kv_group_tokens * q_rows * head_dim;
        trace.executed.fetch_add(1, std::memory_order_relaxed);
        trace.groups.fetch_add(groups, std::memory_order_relaxed);
        trace.kv_group_tokens.fetch_add(kv_group_tokens, std::memory_order_relaxed);
        trace.attention_macs.fetch_add(attention_macs, std::memory_order_relaxed);
        if (kernel_version == AKV_ATTENTION_KERNEL_VERSION_V2) {
            trace.executed_v2.fetch_add(1, std::memory_order_relaxed);
            trace.groups_v2.fetch_add(groups, std::memory_order_relaxed);
        } else {
            trace.executed_v1.fetch_add(1, std::memory_order_relaxed);
            trace.groups_v1.fetch_add(groups, std::memory_order_relaxed);
        }
        std::fprintf(stderr,
                     "GGML_RISCV_AKV_EXEC kernel=v%u kv_heads=%llu q_rows=%llu gqa_rows=%u head_dim=%u "
                     "active_kv=%d attention_macs=%llu\n",
                     kernel_version,
                     static_cast<unsigned long long>(groups),
                     static_cast<unsigned long long>(groups * q_rows), q_rows, head_dim, active_kv,
                     static_cast<unsigned long long>(attention_macs));
    }
    return true;
}
