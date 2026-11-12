#ifndef GGML_AKV_STAGE5_ENTRY_H
#define GGML_AKV_STAGE5_ENTRY_H
#include "platform.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Scalar companion to the sealed v1 API. No tensor access, INFO, lease or CSR
 * write. KEEP_NATIVE here means ordinary GGML, not permission for native AKV. */
akv_platform_result_t ggml_akv_platform_entry_v1(const akv_resident_platform_provider_v1 *);
enum { GGML_AKV_LEGACY_BASE=1, GGML_AKV_LEGACY_PREFILL=2, GGML_AKV_LEGACY_QUANT_KV=4 };
typedef struct {
    unsigned abi_version,struct_bytes;
    const akv_resident_platform_provider_v1 *provider;
    uint8_t contract_sha256[32];
    unsigned operation_mask;
    /* Explicitly validated normal RAM for SDK's internal stack/heap descriptor,
     * query/dequant scratch and workspace, including allocations during call. */
    unsigned whole_process_ram;
} ggml_akv_legacy_contract_v1;
/* Separate explicit registration: resident profile never implies legacy INFO,
 * native kernels or quantKV support. Weak default returns NULL. */
const ggml_akv_legacy_contract_v1 *ggml_akv_legacy_current_v1(void);
int ggml_akv_legacy_permitted_v1(const akv_resident_platform_provider_v1 *,unsigned operations);
akv_platform_result_t ggml_akv_legacy_admit_v1(const akv_resident_platform_provider_v1 *,
    const akv_platform_request_v1 *,akv_platform_session_v1 *,unsigned operations);
typedef struct {
    uint64_t denied, unsupported, lease_denied, resident_calls, native_calls;
    uint64_t resident_groups, native_groups, completed, failed;
    uint64_t legacy_calls,legacy_completed,legacy_fallback;
} ggml_akv_resident_counters_v1;
/* Cumulative control-path counters, not instruction retirement or timing. */
void ggml_akv_resident_snapshot_v1(ggml_akv_resident_counters_v1 *);
#ifdef __cplusplus
}
#endif
#endif
