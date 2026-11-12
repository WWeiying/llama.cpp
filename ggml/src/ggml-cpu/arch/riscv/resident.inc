// Included inside akv.cpp's namespace, only in an opt-in build.
struct resident_group_v1 {
    alignas(64) std::array<ggml_fp16_t, AKV_MAX_Q_ROWS * AKV_HEAD_DIM_128> query;
    alignas(AKV_DESCRIPTOR_BYTES) akv_attention_plan_t plan;
    alignas(AKV_DESCRIPTOR_BYTES) akv_attention_v2_workspace_t workspace;
    akv_attention_features_t features;
    akv_resident_attention_options_t options;
    int64_t first_head;
};

struct resident_counts_v1 {
    std::atomic<uint64_t> denied{}, unsupported{}, lease_denied{}, resident_calls{}, native_calls{};
    std::atomic<uint64_t> resident_groups{}, native_groups{}, completed{}, failed{};
    std::atomic<uint64_t> legacy_calls{},legacy_completed{},legacy_fallback{};
};
static resident_counts_v1 resident_counts;

static bool stage5_decline() {
    resident_counts.unsupported.fetch_add(1, std::memory_order_relaxed);
    return false;
}

static bool stage5_legacy(const ggml_compute_params *params,ggml_tensor *dst,
                          const akv_resident_platform_provider_v1 *provider) {
    const ggml_tensor *q=dst->src[0],*k=dst->src[1];
    if (!q || !k || q->ne[1]<=0 || q->ne[1]>UINT32_MAX || k->ne[2]<=0 || k->ne[2]>UINT32_MAX)
        return stage5_decline();
    unsigned operations=GGML_AKV_LEGACY_BASE;
    if (q->ne[1]>1) operations|=GGML_AKV_LEGACY_PREFILL;
    if (k->type==GGML_TYPE_Q4_0 || k->type==GGML_TYPE_Q8_0) operations|=GGML_AKV_LEGACY_QUANT_KV;
    if (!ggml_akv_legacy_permitted_v1(provider,operations)) return stage5_decline();
    akv_platform_span_v1 spans[6];size_t count=0;
    for (unsigned i=0;i<5;++i) {
        const ggml_tensor *t=dst->src[i];
        if (t) spans[count++]={t->data,ggml_nbytes(t),0};
    }
    spans[count++]={dst->data,ggml_nbytes(dst),1};
    const uintptr_t output=(uintptr_t)dst->data;
    if (spans[count-1].bytes>UINTPTR_MAX-output) return stage5_decline();
    for (size_t i=0;i+1<count;++i) {
        const uintptr_t base=(uintptr_t)spans[i].base;
        if (spans[i].bytes>UINTPTR_MAX-base ||
            (output<base+spans[i].bytes && base<output+spans[count-1].bytes)) return stage5_decline();
    }
    // Legacy contract explicitly approves internal SDK stack/heap allocations.
    // groups here are logical KV groups, NOT hardware command/work counts.
    const akv_platform_request_v1 request={1,sizeof(request),1,unsigned(params->nth),unsigned(q->ne[1]==1),
        uint32_t(q->ne[1]),unsigned(k->ne[2]),0,spans,count};
    akv_platform_session_v1 session{};
    const auto admission=ggml_akv_legacy_admit_v1(provider,&request,&session,operations);
    if (admission<0) GGML_ABORT("AKV legacy platform state: %d",int(admission));
    if (admission!=AKV_PLATFORM_READY) {
        resident_counts.lease_denied.fetch_add(1,std::memory_order_relaxed);return false;
    }
    struct lease_guard {
        akv_platform_session_v1 *session;
        ~lease_guard() { akv_resident_platform_finish_v1(session); }
    } lease{&session};
    // Original body retains its own mutex; do not acquire it twice here.
    resident_counts.legacy_calls.fetch_add(1,std::memory_order_relaxed);
    const bool complete=stage5_legacy_body(params,dst);
    (complete?resident_counts.legacy_completed:resident_counts.legacy_fallback).fetch_add(1,std::memory_order_relaxed);
    return complete;
}

static bool stage5_native(const ggml_compute_params *params, ggml_tensor *dst,
                         const akv_resident_platform_provider_v1 *provider) {
    // Entry CSR guard ran before getenv, tensor access, query conversion or INFO.
    if (!runtime_enabled() || params->nth!=1 || params->ith!=0 || !dst->src[0] || !dst->src[1])
        return stage5_decline();
    if (!ggml_akv_legacy_permitted_v1(provider,GGML_AKV_LEGACY_BASE)) return stage5_decline();
    if (dst->src[0]->ne[1]!=1 || dst->src[1]->type!=GGML_TYPE_F16)
        return stage5_legacy(params,dst,provider);
    int active=0;
    uint32_t version=0,rows=0,dim=0;
    bool portable=false;
    if (supports_decode(params,dst,&active,&version,&rows,&dim,&portable)!=support_reason::supported ||
        version!=AKV_ATTENTION_KERNEL_VERSION_V2) return stage5_legacy(params,dst,provider);
    // Keep the original auto/explicit policy, including an explicit environment
    // override. Never use a Stage4 features resident entry.
    unsigned automatic=0;
#if defined(AKV_NATIVE_AUTO_PAIR_POLICY) && AKV_NATIVE_AUTO_PAIR_POLICY
    automatic=std::getenv("AKV_PAIR_HEADS")==nullptr;
#endif
    const bool requested=ggml_riscv_accel_resident_requested(dim,rows,unsigned(active));
    const ggml_tensor *q=dst->src[0],*k=dst->src[1],*v=dst->src[2],*mask=dst->src[3];
    const uint32_t limit=portable ? device().capabilities.max_q_rows : rows;
    const size_t per_kv=(rows+limit-1)/limit;
    size_t count;
    if (__builtin_mul_overflow(per_kv,size_t(k->ne[2]),&count) || !count || count>UINT32_MAX ||
        count>std::numeric_limits<size_t>::max()/sizeof(resident_group_v1)) return stage5_decline();
    std::vector<resident_group_v1> groups(count);
    akv_platform_span_v1 spans[7]={{q->data,ggml_nbytes(q),0},{k->data,ggml_nbytes(k),0},
        {v->data,ggml_nbytes(v),0},{mask->data,ggml_nbytes(mask),0},
        {dst->data,ggml_nbytes(dst),1},{groups.data(),groups.size()*sizeof(groups[0]),1}};
    size_t span_count=6;
    if (dst->src[4]) spans[span_count++]={dst->src[4]->data,ggml_nbytes(dst->src[4]),0};
    // Writable output must not alias any input, nor the private query/workspace.
    const uintptr_t output=(uintptr_t)spans[4].base;
    if (spans[4].bytes>UINTPTR_MAX-output) return stage5_decline();
    for (size_t i=0;i<span_count;++i) {
        if (i==4) continue;
        const uintptr_t base=(uintptr_t)spans[i].base;
        if (spans[i].bytes>UINTPTR_MAX-base ||
            (output<base+spans[i].bytes && base<output+spans[4].bytes)) return stage5_decline();
    }
    const akv_platform_request_v1 request={AKV_PLATFORM_ABI_V1,sizeof(request),1,1,1,1,
        static_cast<unsigned>(count),0,spans,span_count};
    akv_platform_session_v1 session{};
    auto admission=akv_resident_platform_admit_v1(provider,&request,&session);
    if (admission!=AKV_PLATFORM_READY) {
        if (admission<0) GGML_ABORT("AKV platform state changed before admission: %d",int(admission));
        resident_counts.lease_denied.fetch_add(1,std::memory_order_relaxed);
        return false;
    }
    struct lease_guard {
        akv_platform_session_v1 *session;
        ~lease_guard() { akv_resident_platform_finish_v1(session); }
    } lease{&session};
    std::lock_guard<std::mutex> lock(context_mutex());
    float scale;
    std::memcpy(&scale,dst->op_params,sizeof(scale));
    bool shared=requested;
    size_t index=0;
    for (int64_t kv=0;kv<k->ne[2];++kv) {
        for (uint32_t start=0;start<rows;) {
            auto &g=groups[index];
            const uint32_t n=std::min(rows-start,limit);
            g.first_head=kv*rows+start;
            const akv_attention_problem_t problem={reinterpret_cast<const uint16_t *>(g.query.data()),
                reinterpret_cast<const uint16_t *>(static_cast<const char *>(k->data)+kv*k->nb[2]),
                reinterpret_cast<const uint16_t *>(static_cast<const char *>(v->data)+kv*v->nb[2]),
                static_cast<const uint16_t *>(mask->data),
                reinterpret_cast<float *>(static_cast<char *>(dst->data)+g.first_head*dst->nb[1]),
                dim*sizeof(ggml_fp16_t),k->nb[1],v->nb[1],dst->nb[1],n,dim,uint32_t(active),scale};
            if (akv_attention_plan_create_v2(&device(),&problem,&g.plan)!=AKV_STATUS_OK)
                return stage5_decline(); // no query FP, output, workspace or FULL yet
            if (portable && !decode_features(dst,uint32_t(g.first_head),n,active,&g.features))
                return stage5_decline();
            if (shared) {
                admission=akv_resident_platform_group_v1(&session,unsigned(index),&device(),
                    &g.plan,portable?&g.features:nullptr,&g.options);
                if (admission<0) GGML_ABORT("AKV platform preflight state: %d",int(admission));
                shared=admission==AKV_PLATFORM_READY;
            }
            start+=n;
            ++index;
        }
    }
    // Re-read real state for either route. An unsupported shape/feature keeps
    // the ORIGINAL native route, but only under this same trusted lease.
    if (ggml_akv_platform_entry_v1(provider)!=AKV_PLATFORM_READY)
        GGML_ABORT("AKV platform state changed before execution");
    if (shared && akv_resident_platform_start_v1(&session)!=AKV_PLATFORM_READY)
        GGML_ABORT("AKV platform whole-call start failed");
    (shared?resident_counts.resident_calls:resident_counts.native_calls).fetch_add(1,std::memory_order_relaxed);
    const ggml_from_float_t convert=ggml_get_type_traits_cpu(GGML_TYPE_F16)->from_float;
    for (auto &g:groups) {
        for (uint32_t h=0;h<g.plan.descriptor.q_rows;++h) {
            const float *source=reinterpret_cast<const float *>(static_cast<const char *>(q->data)+(g.first_head+h)*q->nb[2]);
            convert(source,g.query.data()+h*dim,dim);
        }
        const akv_status_t status=shared ? akv_experimental_attention_execute_v2_native(&device(),
            &g.plan,&g.workspace,portable?&g.features:nullptr,&g.options,automatic)
            : (portable && !akv_attention_can_use_plain_scores(&g.features,g.plan.mask,active)
                ? execute_features_native(&g.plan,&g.workspace,&g.features) : execute(&g.plan,&g.workspace));
        if (status!=AKV_STATUS_OK) {
            resident_counts.failed.fetch_add(1,std::memory_order_relaxed);
            GGML_ABORT("AKV execution failed after route fixation: %s",akv_status_string(status));
        }
        (shared?resident_counts.resident_groups:resident_counts.native_groups).fetch_add(1,std::memory_order_relaxed);
    }
    resident_counts.completed.fetch_add(1,std::memory_order_relaxed);
    if (env_true("GGML_RISCV_AKV_TRACE")) {
        register_trace();
        trace.candidates.fetch_add(1,std::memory_order_relaxed);
        trace.executed.fetch_add(1,std::memory_order_relaxed);
        trace.executed_v2.fetch_add(1,std::memory_order_relaxed);
        trace.executed_decode.fetch_add(1,std::memory_order_relaxed);
        trace.groups.fetch_add(count,std::memory_order_relaxed);
        trace.groups_v2.fetch_add(count,std::memory_order_relaxed);
        trace.kv_group_tokens.fetch_add(count*active,std::memory_order_relaxed);
        trace.attention_macs.fetch_add(UINT64_C(2)*k->ne[2]*active*rows*dim,std::memory_order_relaxed);
        std::fprintf(stderr,"GGML_RISCV_AKV_RESIDENT_V1 route=%s requested=%u groups=%zu dimension=%u tokens=%d\n",
            shared?"resident":"native",unsigned(requested),count,dim,active);
    }
    return true;
}
