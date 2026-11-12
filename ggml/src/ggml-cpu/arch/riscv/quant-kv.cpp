#include "quant-kv.h"
#include "accel-options.h"

#include "qbs.h"
#include "../../ggml-cpu-impl.h"
#include "../../vec.h"
#include "ggml_entry_strided.h"
#include "q8_0_value.h"
#ifdef GGML_USE_RISCV_REGISTER_PV
#include "register_pv/ggml_entry.h"
#endif

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef GGML_USE_RISCV_QUANT_PREFILL
#include "prefill/ggml_entry.h"
#include <memory>
#endif

namespace {

static bool enabled(const char * name) {
    const char * value = ggml_riscv_accel_option(name);
    return value && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0 || std::strcmp(value, "YES") == 0 || std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0);
}

static void decode_value(const void * source, float * output, int64_t elements) {
    akv_quant_q8_0_value(source, output, static_cast<size_t>(elements));
}

static void scale_output(int n, float * output, float scale) {
    ggml_vec_scale_f32(n, output, scale);
}

static void accumulate_value(int n, float * output, const float * value, float scale) {
    ggml_vec_mad_f32(n, output, value, scale);
}

struct invocation {
    const ggml_tensor * dst;
    akv_quant_ggml_context_t context;
    akv_quant_native_bindings_t bindings;
    akv_quant_ggml_strided_report_t report{};
    void * workspace;
    size_t workspace_bytes;
    bool grouped_pv;
    bool register_pv;
};

static int execute(void * opaque, qbs_command_executor_t executor) {
    auto & call = *static_cast<invocation *>(opaque);
    call.bindings.execute = executor;
#ifdef GGML_USE_RISCV_REGISTER_PV
    if (call.register_pv) {
        return quant_register_ggml_execute(call.dst, &call.context, &call.bindings,
            call.workspace, call.workspace_bytes, &call.report);
    }
#endif
    if (call.grouped_pv) {
        return akv_quant_ggml_strided_execute_grouped_pv(call.dst, &call.context, &call.bindings, call.workspace, call.workspace_bytes, &call.report);
    }
    return akv_quant_ggml_strided_execute(call.dst, &call.context, &call.bindings, call.workspace, call.workspace_bytes, &call.report);
}

#ifdef GGML_USE_RISCV_QUANT_PREFILL
struct prefill_invocation {
    const ggml_tensor * dst;
    akv_quant_ggml_context_t context;
    akv_quant_native_bindings_t bindings;
    quant_prefill_ggml_report_t report{};
    void * workspace;
    size_t bytes;
};

static int execute_prefill(void * opaque, qbs_command_executor_t executor) {
    auto & call = *static_cast<prefill_invocation *>(opaque);
    call.bindings.execute = executor;
    return quant_prefill_ggml_execute(call.dst, &call.context, &call.bindings,
                                      call.workspace, call.bytes, &call.report);
}

static ggml_riscv_quant_kv_status run_prefill(const ggml_compute_params * params,
        ggml_tensor * dst, const qbs_device_t * device, akv_quant_native_bindings_t bindings) {
    if (!enabled("GGML_RISCV_AKV_QUANT_PREFILL") || !device) {
        return ggml_riscv_quant_kv_status::fallback;
    }
    const size_t bytes = quant_prefill_ggml_workspace_bytes();
    if (quant_prefill_ggml_workspace_alignment() != 64 || bytes > SIZE_MAX - 63) {
        return ggml_riscv_quant_kv_status::fallback;
    }
    std::unique_ptr<unsigned char, decltype(&std::free)> allocation(
        static_cast<unsigned char *>(std::malloc(bytes + 63)), &std::free);
    if (!allocation) { return ggml_riscv_quant_kv_status::fallback; }
    uintptr_t vlenb, vstart, frm;
    __asm__ volatile("csrr %0,vlenb;csrr %1,vstart;csrr %2,frm"
                     : "=r"(vlenb), "=r"(vstart), "=r"(frm) : : "memory");
    prefill_invocation call{};
    call.dst = dst;
    call.context.requested = 1;
    call.context.ith = params->ith;
    call.context.nth = params->nth;
    call.context.use_ref = params->use_ref;
    call.context.vlen_bits = static_cast<unsigned>(vlenb * 8);
    call.context.vstart = static_cast<unsigned>(vstart);
    call.context.frm = static_cast<unsigned>(frm);
    call.context.normal_memory = 1;
    call.context.device = device;
    call.bindings = bindings;
    call.workspace = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(allocation.get()) + 63) & ~uintptr_t(63));
    call.bytes = bytes;
    const int status = ggml_riscv_qbs_direct_run(execute_prefill, &call);
    if (enabled("GGML_RISCV_AKV_TRACE")) {
        std::fprintf(stderr, "GGML_RISCV_QUANT_PREFILL status=%d mapping=%u memory=%u runtime_called=%u selection=%u commands=%u groups=%u value_decodes=%u queries_quantized=%u key_bytes=%zu workspace=%zu\n",
            status, call.report.mapping_reason, call.report.memory_verified, call.report.runtime_called,
            call.report.native.selection_reason, call.report.native.command_attempts,
            call.report.native.completed_groups, call.report.native.value_decodes,
            call.report.queries_quantized, call.report.key_bytes, bytes);
    }
    if (status == AKV_QUANT_NATIVE_OK) { return ggml_riscv_quant_kv_status::complete; }
    return status == AKV_QUANT_NATIVE_FALLBACK ? ggml_riscv_quant_kv_status::fallback
                                             : ggml_riscv_quant_kv_status::failed;
}
#endif

} // namespace

ggml_riscv_quant_kv_status ggml_riscv_quant_kv_flash_attn(const ggml_compute_params * params, ggml_tensor * dst) {
    if (!params || !dst || params->nth != 1 || params->ith != 0 || params->use_ref || !enabled("GGML_RISCV_AKV_QUANT_KV_NORMAL_MEMORY")) {
        return ggml_riscv_quant_kv_status::fallback;
    }
    const ggml_tensor * key = dst->src[1];
    if (!key || (key->type != GGML_TYPE_Q4_0 && key->type != GGML_TYPE_Q8_0)) {
        return ggml_riscv_quant_kv_status::fallback;
    }
    const auto * key_traits = ggml_get_type_traits_cpu(key->type);
    if (!key_traits || key_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        return ggml_riscv_quant_kv_status::fallback;
    }
    const auto * query_traits = ggml_get_type_traits_cpu(key_traits->vec_dot_type);
    if (!query_traits || !query_traits->from_float) {
        return ggml_riscv_quant_kv_status::fallback;
    }
    const qbs_device_t * device = ggml_riscv_qbs_direct_device();
#ifdef GGML_USE_RISCV_QUANT_PREFILL
    if (dst->src[0] && dst->src[0]->ne[1] > 1) {
        akv_quant_native_bindings_t bindings{};
        bindings.quantize = query_traits->from_float;
        bindings.decode = decode_value;
        bindings.scale = scale_output;
        bindings.mad = accumulate_value;
        return run_prefill(params, dst, device, bindings);
    }
#endif
    const size_t bytes = akv_quant_strided_workspace_bytes();
    alignas(64) std::array<unsigned char, 16384> workspace;
    if (!device || bytes > workspace.size() || akv_quant_native_workspace_alignment() != 64) {
        return ggml_riscv_quant_kv_status::fallback;
    }
    uintptr_t vlenb, vstart, frm;
    __asm__ volatile("csrr %0,vlenb;csrr %1,vstart;csrr %2,frm" : "=r"(vlenb), "=r"(vstart), "=r"(frm) : : "memory");
    invocation call{};
    call.dst = dst;
    call.context.requested = 1;
    call.context.ith = params->ith;
    call.context.nth = params->nth;
    call.context.vlen_bits = static_cast<unsigned>(vlenb * 8);
    call.context.vstart = static_cast<unsigned>(vstart);
    call.context.frm = static_cast<unsigned>(frm);
    call.context.normal_memory = 1;
    call.context.device = device;
    call.bindings.quantize = query_traits->from_float;
    call.bindings.decode = decode_value;
    call.bindings.scale = scale_output;
    call.bindings.mad = accumulate_value;
    call.workspace = workspace.data();
    call.workspace_bytes = bytes;
#if defined(GGML_SIMD) && defined(__riscv_v_intrinsic) && defined(__riscv_vector)
    call.grouped_pv = enabled("GGML_RISCV_AKV_GROUPED_PV");
#endif
#ifdef GGML_USE_RISCV_REGISTER_PV
    call.register_pv = enabled("GGML_RISCV_AKV_REGISTER_PV");
    // Keep experiments mutually exclusive; neither choice changes Prefill.
    if (call.register_pv && call.grouped_pv) {
        return ggml_riscv_quant_kv_status::fallback;
    }
#endif
    const int status = ggml_riscv_qbs_direct_run(execute, &call);
    if (enabled("GGML_RISCV_AKV_TRACE")) {
        const auto & native = call.report.execution.native;
        std::fprintf(stderr, "GGML_RISCV_QUANT_KV status=%d mapping=%u memory=%u runtime_called=%u command_attempts=%u groups=%u value_decodes=%u executor_status=%u\n", status, call.report.mapping_reason, call.report.memory_verified, call.report.runtime_called, native.command_attempts, native.completed_groups, native.value_decodes, static_cast<unsigned>(native.executor_status));
        std::fprintf(stderr, "GGML_RISCV_QUANT_KV_LAYOUT selection=%u key_panels=%zu key_bytes=%zu workspace=%zu k_token_stride=%zu k_head_stride=%zu\n", native.selection_reason, call.report.execution.key_panels, call.report.execution.key_bytes, bytes, key->nb[1], key->nb[2]);
        std::fprintf(stderr, "GGML_RISCV_QUANT_KV_PV grouped=%u completed_groups=%u\n", unsigned(call.grouped_pv), native.completed_groups);
        std::fprintf(stderr, "GGML_RISCV_QUANT_KV_REGISTER_PV enabled=%u status=%d completed_groups=%u\n", unsigned(call.register_pv), status, native.completed_groups);
    }
    if (status == AKV_QUANT_NATIVE_OK) {
        return ggml_riscv_quant_kv_status::complete;
    }
    return status == AKV_QUANT_NATIVE_FALLBACK ? ggml_riscv_quant_kv_status::fallback : ggml_riscv_quant_kv_status::failed;
}
