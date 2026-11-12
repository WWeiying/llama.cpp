#ifndef AKV_EXPERIMENTAL_PLATFORM_V1_H
#define AKV_EXPERIMENTAL_PLATFORM_V1_H

#include <stddef.h>
#include <stdint.h>
#include "resident_attention/admission.h"
#include "akv/akv_features.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { AKV_PLATFORM_ABI_V1=1, AKV_PLATFORM_QK_PV_V1=3 };
enum {
  AKV_PLATFORM_VALIDATED_TEST_MODEL=1,
  AKV_PLATFORM_EXTERNAL_DEPLOYMENT=2
};
enum {
  AKV_PLATFORM_BOUND_LAUNCHER=1u<<0,
  AKV_PLATFORM_BOUND_IMAGE_ISA=1u<<1,
  AKV_PLATFORM_BOUND_MEMORY=1u<<2,
  AKV_PLATFORM_EXCLUSIVE_PROCESS=1u<<3,
  AKV_PLATFORM_SINGLE_WORKER=1u<<4,
  AKV_PLATFORM_SAFE_STATE_ACCESS=1u<<5,
  AKV_PLATFORM_VALIDATED_INSTRUCTIONS=1u<<6,
  AKV_PLATFORM_REQUIRED_ASSURANCES=(1u<<7)-1
};
typedef enum {
  AKV_PLATFORM_KEEP_NATIVE=0,
  AKV_PLATFORM_READY=1,
  AKV_PLATFORM_BAD_VSTART=-1,
  AKV_PLATFORM_INVALID=-2,
  AKV_PLATFORM_NO_RETRY=-3
} akv_platform_result_t;

typedef struct {
  uintptr_t vlenb,vstart,fcsr,vcsr;
} akv_platform_thread_state_v1;

typedef struct {
  const void *base;
  size_t bytes;
  unsigned writable;
} akv_platform_span_v1;

typedef struct {
  unsigned abi_version,struct_bytes;
  unsigned requested,workers,decode_head_rows,query_tokens,groups,refill_overlap;
  /* Q, K, V, mask, destination, query scratch and workspace; input Q may be F32.
   * The provider validates every accessible span/lifetime, not just endpoints. */
  const akv_platform_span_v1 *spans;
  size_t span_count;
} akv_platform_request_v1;

typedef struct akv_resident_platform_provider_v1 {
  unsigned abi_version,struct_bytes,kind,assurances;
  /* SHA256 of the approved launcher+image/model+ISA+OS+RAM/ownership contract.
   * This is provenance supplied by trusted code, not an application credential. */
  uint8_t contract_sha256[32];
  unsigned operation_mask,qk_profile,pv_profile,nr_lanes,vlen_bits;
  void *context;
  /* Must be scalar/trap-safe on this thread before any request/device copy.
   * No FP/vector instruction, CSR write, custom instruction or libc copy. */
  akv_platform_result_t (*read_state)(void *,akv_platform_thread_state_v1 *);
  /* Nonzero cookie affirms ALL spans, process/worker/OS binding and exclusive
   * ownership through SDK RELEASE. Failed acquisition must retain no lease.
   * Both callbacks preserve FCSR/VCSR/VSTART and never execute accelerator work.
   * release is scalar-safe even following a BAD_VSTART rejection. */
  uintptr_t (*acquire)(void *,const akv_platform_request_v1 *);
  void (*release)(void *,uintptr_t);
} akv_resident_platform_provider_v1;

typedef struct {
  const akv_resident_platform_provider_v1 *provider;
  uintptr_t cookie;
  unsigned groups,prepared,started;
  const akv_platform_span_v1 *spans;
  size_t span_count;
} akv_platform_session_v1;

/* Default weak implementation returns NULL. Registration is an explicit linked
 * platform implementation, never an environment string or generic setter. */
const akv_resident_platform_provider_v1 *akv_resident_platform_current_v1(void);

/* Failure leaves session/options unchanged. Initialize session to all zero.
 * KEEP_NATIVE is a resident decline, NOT permission to issue legacy INFO/AKV.
 * Without independent native admission, the frontend uses ordinary GGML/RVV.
 * VSTART rejection is not permission to enter an ordinary vector fallback. */
akv_platform_result_t akv_resident_platform_admit_v1(
    const akv_resident_platform_provider_v1 *,const akv_platform_request_v1 *,
    akv_platform_session_v1 *);
akv_platform_result_t akv_resident_platform_group_v1(
    akv_platform_session_v1 *,unsigned group,const akv_device_t *,
    const akv_attention_plan_t *,const akv_attention_features_t *,
    akv_resident_attention_options_t *);
akv_platform_result_t akv_resident_platform_start_v1(akv_platform_session_v1 *);
void akv_resident_platform_finish_v1(akv_platform_session_v1 *);

/* RV64 scalar implementation in read_state.S. Call only after provider's
 * OS/privilege-safe state-access contract is established. No machine CSR read. */
akv_platform_result_t akv_resident_platform_read_state_v1(
    void *,akv_platform_thread_state_v1 *);

#ifdef __cplusplus
}
#endif
#endif
