#pragma once

#include "q8_K-prepared-cache.h"
#include "m4-reuse-policy.h"
#include "ggml-cpu-impl.h"
#include "arch/riscv/qbs.h"
#include "q8_K-m4-packed.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_riscv_q8_K_prepared {

inline bool enabled() {
    static const bool value=[] {
        const char *s=std::getenv("GGML_RISCV_QBS_M4_CACHE");
        return s && std::strcmp(s,"1")==0;
    }();
    return value;
}
inline bool supported(const ggml_tensor *node) {
    if(!node || !node->src[0] || !node->src[1] || !node->src[0]->extra) return false;
    const auto *w=node->src[0];const auto *x=node->src[1];
    switch(w->type) {
        case GGML_TYPE_Q2_K:case GGML_TYPE_Q3_K:case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:case GGML_TYPE_Q6_K:break;
        default:return false;
    }
    ggml_riscv_qbs_activation_plan plan{};
    return ggml_riscv_qbs_supports_tensor(w->type,x->ne[0],w->ne[1]) &&
        ggml_riscv_qbs_plan_activations(w->type,x->ne[1],w->ne[1],x->ne[0],&plan) && !plan.wide_m;
}
struct state {
    ggml_riscv_q8_K_cache::value_cache<> bytes;
    uint64_t epoch=0,computations=0,generic_computations=0;
    const ggml_cgraph *graph=nullptr;
    const ggml_tensor *source=nullptr;
    ggml_riscv_m4_reuse::policy begin(const ggml_compute_params *params,const ggml_tensor *op) {
        ggml_riscv_m4_reuse::policy policy{};
        if(params && params->nth==1 && params->graph_epoch && params->cgraph && op)
            policy=ggml_riscv_m4_reuse::select(params->cgraph,params->node_index,op,supported);
        const auto *input=op ? op->src[1]:nullptr;
        if(!params || graph!=params->cgraph || epoch!=params->graph_epoch || source!=input || !policy.lookup)
            bytes.clear();
        graph=params ? params->cgraph:nullptr;epoch=params ? params->graph_epoch:0;source=input;
        return policy;
    }
    // True means the output was produced, including a generic producer fallback.
    // Allocation failure must never cause a second quantization of that output.
    bool prepare(ggml_riscv_m4_reuse::policy policy,const float *input,void *output,
                 int64_t k,int64_t group,bool trace) {
        if(!policy.lookup || group<0 || group>=8 || k<=0 || k%256 || k>6144) return false;
        unsigned before,start;
        asm volatile("frcsr %0;csrr %1,vstart":"=r"(before),"=r"(start)::"memory");
        if((before>>5)!=0 || start) return false;
        ggml_riscv_q8_K_cache::key identity;identity.k=uint32_t(k);identity.producer_tag=ggml_riscv_q8_K_m4_packed::producer_tag();
        const size_t in=size_t(k)*16,out=size_t(k/256)*4*QBS_Q8_K_BLOCK_BYTES;
        unsigned raised=0;const char *event="hit";
        if(!bytes.get(identity,input,in,output,out,raised)) {
            asm volatile("fsflags zero":::"memory");
            const bool vector=ggml_riscv_q8_K_m4_packed::selected_quantize(input,output,k);
            asm volatile("frflags %0":"=r"(raised)::"memory");
            const unsigned restored=(before&31)|raised;
            asm volatile("fsflags %0"::"r"(restored):"memory");
            ++computations;generic_computations+=!vector;
            if(vector && policy.fill) (void)bytes.put(identity,input,in,output,out,raised);
            event=vector ? "compute_rvv":"compute_generic";
        }
        const unsigned restored=(before&31)|raised;
        asm volatile("fsflags %0"::"r"(restored):"memory");
        if(trace) std::fprintf(stderr,"GGML_RISCV_QBS_M4_CACHE epoch=%llu group=%lld event=%s fill=%d\n",
            (unsigned long long)epoch,(long long)group,event,int(policy.fill));
        if(trace) std::fprintf(stderr,"GGML_RISCV_QBS_M4_SELECTED source=cache tag=%u event=%s k=%lld\n",
            identity.producer_tag,event,(long long)k);
        return true;
    }
    void finish(ggml_riscv_m4_reuse::policy policy) {if(!policy.fill) bytes.clear();}
};
inline state &worker_state() {static thread_local state value;return value;}

} // namespace ggml_riscv_q8_K_prepared
