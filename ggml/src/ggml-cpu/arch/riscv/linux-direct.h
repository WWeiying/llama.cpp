#ifndef GGML_RISCV_LINUX_DIRECT_H
#define GGML_RISCV_LINUX_DIRECT_H

#include "entry.h"

namespace ggml_akv_direct {

struct span { const void * base; size_t bytes; bool writable; };

// Scalar OS/ISA checks precede CSR access and ordinary F/V initialization.
// Nonzero VSTART is fatal, not permission to run an RVV fallback.
// Bootstrap once at launcher startup with VSTART=0; syscalls can discard the
// pre-probe vector state. Keep F/V policy unchanged after successful admission.
bool available();

// Allocate all storage first. No allocation, mapping change, thread creation,
// logging or lazy initialization between successful prepare() and end().
// The fixed launcher must keep file mappings immutable and use one thread.
// MCL_CURRENT|MCL_FUTURE remains installed for this process after preparation.
// maps identity caches the smaps audit, NOT independently observed lock state.
// Do not change mlock/madvise/userfault policy behind this helper. Invalidate
// before an allowed out-of-lease policy change; no such change is allowed live.
bool prepare(const span *, size_t);
bool begin();
bool end(); // Release failure is fatal. False means no active lease.
bool active();
const akv_resident_platform_provider_v1 * provider();
const ggml_akv_legacy_contract_v1 * legacy_contract();
void invalidate();
const char * last_error();
struct statistics {
    uint64_t prepares, maps_reads, smaps_reads, cache_hits, lock_calls, acquires, releases;
};
statistics stats();

// Compile with GGML_AKV_DIRECT_CONTRACT_SHA256="<64 hex digits>" from the
// reviewed v4 kernel/image/ISA/RAM/launcher contract. No pin disables admission.
// GET_INFO alone cannot attest the kernel revision or arithmetic capability.
constexpr size_t stack_reserve_bytes = 512 * 1024;
constexpr size_t max_spans = 32;

}
#endif
