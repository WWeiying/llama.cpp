#ifndef GGML_RISCV_Q8_K_M4_GROUP_SUMS_H
#define GGML_RISCV_Q8_K_M4_GROUP_SUMS_H
#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

/* Internal leaf: vl is positive, divisible by 16, and fits e8m1 VLMAX.
 * Destination has one writable i16 every four i16s, as in block_q8_Kx4.
 * Full-width additions may cross group boundaries at unselected lanes.
 * Only positions 0,16,... are gathered, each summing exactly its 16 bytes.
 * Every partial sum fits i16: the final range is [-2048,2032]. */
static inline void ggml_riscv_q8_K_m4_group_sums(vint8m1_t q,int16_t *out,size_t vl) {
  vint16m2_t sum=__riscv_vsext_vf2_i16m2(q,vl);
  sum=__riscv_vadd_vv_i16m2(sum,__riscv_vslidedown_vx_i16m2(sum,8,vl),vl);
  sum=__riscv_vadd_vv_i16m2(sum,__riscv_vslidedown_vx_i16m2(sum,4,vl),vl);
  sum=__riscv_vadd_vv_i16m2(sum,__riscv_vslidedown_vx_i16m2(sum,2,vl),vl);
  sum=__riscv_vadd_vv_i16m2(sum,__riscv_vslidedown_vx_i16m2(sum,1,vl),vl);
  const size_t groups=vl/16;
  vuint16m2_t index=__riscv_vid_v_u16m2(groups);
  index=__riscv_vsll_vx_u16m2(index,4,groups);
  sum=__riscv_vrgather_vv_i16m2(sum,index,groups);
  __riscv_vsse16_v_i16m2(out,4*sizeof(int16_t),sum,groups);
}
#endif
