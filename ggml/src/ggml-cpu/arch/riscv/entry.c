#include "entry.h"

__attribute__((weak))
const ggml_akv_legacy_contract_v1 *ggml_akv_legacy_current_v1(void) { return NULL; }

int ggml_akv_legacy_permitted_v1(const akv_resident_platform_provider_v1 *p,unsigned operations) {
    const ggml_akv_legacy_contract_v1 *c=ggml_akv_legacy_current_v1();
    if (!p || !c || c->abi_version!=1 || c->struct_bytes!=sizeof(*c) || c->provider!=p ||
        c->whole_process_ram!=1 || !operations || operations&~7u ||
        (c->operation_mask&operations)!=operations || c->operation_mask&~7u) return 0;
    const volatile uint8_t *a=p->contract_sha256,*b=c->contract_sha256;
    unsigned difference=0;
    for (size_t i=0;i<32;++i) difference|=a[i]^b[i];
    return difference==0;
}

akv_platform_result_t ggml_akv_platform_entry_v1(const akv_resident_platform_provider_v1 *p) {
    /* Same immutable provider contract as platform.c. The sealed v1 API has
     * no state-only entry: this companion checks before GGML reads tensors. */
    if (!p || p->abi_version!=1 || p->struct_bytes!=sizeof(*p) ||
        (p->kind!=AKV_PLATFORM_VALIDATED_TEST_MODEL && p->kind!=AKV_PLATFORM_EXTERNAL_DEPLOYMENT) ||
        p->assurances!=AKV_PLATFORM_REQUIRED_ASSURANCES || p->operation_mask!=AKV_PLATFORM_QK_PV_V1 ||
        p->qk_profile!=1 || p->pv_profile!=1 || p->nr_lanes!=4 || p->vlen_bits!=1024 ||
        !p->read_state || !p->acquire || !p->release) return AKV_PLATFORM_KEEP_NATIVE;
    unsigned nonzero=0;
    const volatile uint8_t *identity=p->contract_sha256;
    for (size_t i=0;i<sizeof(p->contract_sha256);++i) nonzero|=identity[i];
    if (!nonzero) return AKV_PLATFORM_KEEP_NATIVE;
    akv_platform_thread_state_v1 state;
    const akv_platform_result_t result=p->read_state(p->context,&state);
    if (result==AKV_PLATFORM_BAD_VSTART) return result;
    if (result!=AKV_PLATFORM_READY) return AKV_PLATFORM_KEEP_NATIVE;
    if (state.vstart) return AKV_PLATFORM_BAD_VSTART;
    return state.vlenb==128 && !(state.fcsr>>5) && state.vcsr<=7
        ? AKV_PLATFORM_READY : AKV_PLATFORM_KEEP_NATIVE;
}

akv_platform_result_t ggml_akv_legacy_admit_v1(const akv_resident_platform_provider_v1 *p,
    const akv_platform_request_v1 *r,akv_platform_session_v1 *out,unsigned operations) {
    akv_platform_result_t result=ggml_akv_platform_entry_v1(p);
    if (result!=AKV_PLATFORM_READY) return result;
    if (!ggml_akv_legacy_permitted_v1(p,operations)) return AKV_PLATFORM_KEEP_NATIVE;
    if (!r || !out || out->cookie || r->abi_version!=1 || r->struct_bytes!=sizeof(*r))
        return AKV_PLATFORM_INVALID;
    if (r->requested!=1 || r->workers!=1 || !r->groups || !r->query_tokens ||
        !r->spans || !r->span_count || r->span_count>32) return AKV_PLATFORM_KEEP_NATIVE;
    for (size_t i=0;i<r->span_count;++i) {
        const uintptr_t base=(uintptr_t)r->spans[i].base;
        if (!base || !r->spans[i].bytes || r->spans[i].bytes>UINTPTR_MAX-base || r->spans[i].writable>1)
            return AKV_PLATFORM_KEEP_NATIVE;
    }
    akv_platform_thread_state_v1 before,after;
    result=p->read_state(p->context,&before);
    if (result!=AKV_PLATFORM_READY)
        return result==AKV_PLATFORM_BAD_VSTART ? result : AKV_PLATFORM_KEEP_NATIVE;
    if (before.vstart) return AKV_PLATFORM_BAD_VSTART;
    if (before.fcsr>>5 || before.vlenb!=128 || before.vcsr>7) return AKV_PLATFORM_KEEP_NATIVE;
    const uintptr_t cookie=p->acquire(p->context,r);
    if (!cookie) return AKV_PLATFORM_KEEP_NATIVE;
    result=p->read_state(p->context,&after);
    if (result!=AKV_PLATFORM_READY) {
        p->release(p->context,cookie);
        return result==AKV_PLATFORM_BAD_VSTART ? result : AKV_PLATFORM_INVALID;
    }
    if (after.vstart || after.vlenb!=before.vlenb ||
        after.fcsr!=before.fcsr || after.vcsr!=before.vcsr) {
        p->release(p->context,cookie);
        return after.vstart ? AKV_PLATFORM_BAD_VSTART : AKV_PLATFORM_INVALID;
    }
    out->provider=p;out->cookie=cookie;out->groups=r->groups;out->prepared=0;out->started=1;
    out->spans=r->spans;out->span_count=r->span_count;
    return AKV_PLATFORM_READY;
}
