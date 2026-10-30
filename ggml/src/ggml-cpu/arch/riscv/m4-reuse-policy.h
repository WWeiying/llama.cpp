#pragma once

#include "ggml.h"
#include <cstdint>

namespace ggml_riscv_m4_reuse {

struct policy {
    bool lookup=false;
    bool fill=false;
};
using consumer_supported=bool (*)(const ggml_tensor *);

// A profitability hint only. A hit still requires owned, exact input bytes and
// matching numerical controls; graph identity never establishes cache validity.
inline bool eligible(const ggml_tensor *node,consumer_supported supported) {
    if(!node || !supported || node->op!=GGML_OP_MUL_MAT || !node->src[0] || !node->src[1]) return false;
    const auto *x=node->src[1];
    return x->type==GGML_TYPE_F32 && x->data && x->ne[0]>0 && x->ne[0]%256==0 &&
        x->ne[1]>=4 && x->ne[2]==1 && x->ne[3]==1 && ggml_is_contiguous(x) && supported(node);
}

inline bool may_write(const ggml_tensor *node,const ggml_tensor *input) {
    if(!node || !input) return true;
    switch(node->op) {
        case GGML_OP_NONE:case GGML_OP_VIEW:case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:case GGML_OP_TRANSPOSE:return false;
        case GGML_OP_MAP_CUSTOM1:case GGML_OP_MAP_CUSTOM2:case GGML_OP_MAP_CUSTOM3:
        case GGML_OP_CUSTOM:case GGML_OP_OPT_STEP_ADAMW:case GGML_OP_OPT_STEP_SGD:return true;
        default:break;
    }
    if(!node->data || !input->data) return true;
    const size_t write_bytes=ggml_nbytes(node),input_bytes=ggml_nbytes(input);
    if(!write_bytes || !input_bytes) return false;
    const uintptr_t a=reinterpret_cast<uintptr_t>(node->data),b=reinterpret_cast<uintptr_t>(input->data);
    return a>=b ? a-b<input_bytes:b-a<write_bytes;
}

inline bool neighbor(ggml_cgraph *graph,int index,int direction,
                     const ggml_tensor *input,consumer_supported supported) {
    const int count=ggml_graph_n_nodes(graph);
    // Bound planning overhead; stopping early only disables this optimization.
    for(int i=index+direction,visited=0;i>=0 && i<count && visited<128;i+=direction,++visited) {
        const auto *next=ggml_graph_node(graph,i);
        if(!next || may_write(next,input) || next->op==GGML_OP_MUL_MAT_ID) return false;
        if(next->op!=GGML_OP_MUL_MAT) continue;
        if(eligible(next,supported)) return next->src[1]==input;
        // Non-QBS float matrix work cannot consume this packed representation.
        if(next->src[0] && next->src[1] && next->src[1]->type==GGML_TYPE_F32 &&
           (next->src[0]->type==GGML_TYPE_F32 || next->src[0]->type==GGML_TYPE_F16)) continue;
        return false;
    }
    return false;
}

inline policy select(ggml_cgraph *graph,int index,const ggml_tensor *node,
                     consumer_supported supported) {
    if(!graph || index<0 || index>=ggml_graph_n_nodes(graph) ||
       ggml_graph_node(graph,index)!=node || !eligible(node,supported)) return {};
    const auto *input=node->src[1];
    // Include the current consumer: its result must not overwrite this input.
    if(may_write(node,input)) return {};
    const bool previous=neighbor(graph,index,-1,input,supported);
    const bool next=neighbor(graph,index,1,input,supported);
    return {previous || next,next};
}

} // namespace ggml_riscv_m4_reuse
