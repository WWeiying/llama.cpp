#pragma once

#include <cstddef>
#include <cstring>
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace ggml_riscv_byte_ops {

// Normal RVV C ABI: vstart must be zero. VL/VTYPE/vector registers are clobbered;
// integer nonsaturating operations leave FCSR and VCSR unchanged.
inline bool equal(const void *left,const void *right,size_t bytes) {
#if defined(__riscv_vector)
    const auto *a=static_cast<const unsigned char *>(left);
    const auto *b=static_cast<const unsigned char *>(right);
    for(size_t i=0;i<bytes;) {
        const size_t vl=__riscv_vsetvl_e8m8(bytes-i);
        const auto x=__riscv_vle8_v_u8m8(a+i,vl),y=__riscv_vle8_v_u8m8(b+i,vl);
        const auto mismatch=__riscv_vmsne_vv_u8m8_b1(x,y,vl);
        if(__riscv_vfirst_m_b1(mismatch,vl)>=0) return false;
        i+=vl;
    }
    return true;
#else
    return bytes==0 || std::memcmp(left,right,bytes)==0;
#endif
}

// memcpy semantics: input and output ranges must not overlap.
inline void copy(void *destination,const void *source,size_t bytes) {
#if defined(__riscv_vector)
    auto *d=static_cast<unsigned char *>(destination);
    const auto *s=static_cast<const unsigned char *>(source);
    for(size_t i=0;i<bytes;) {
        const size_t vl=__riscv_vsetvl_e8m8(bytes-i);
        const auto value=__riscv_vle8_v_u8m8(s+i,vl);
        __riscv_vse8_v_u8m8(d+i,value,vl);i+=vl;
    }
#else
    if(bytes) std::memcpy(destination,source,bytes);
#endif
}

} // namespace ggml_riscv_byte_ops
