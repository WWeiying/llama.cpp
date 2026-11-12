#include "platform.h"

__attribute__((weak))
const akv_resident_platform_provider_v1 *akv_resident_platform_current_v1(void) {
  return NULL;
}

static int provider_valid(const akv_resident_platform_provider_v1 *p) {
  if(!p || p->abi_version!=AKV_PLATFORM_ABI_V1 || p->struct_bytes!=sizeof(*p) ||
     (p->kind!=AKV_PLATFORM_VALIDATED_TEST_MODEL && p->kind!=AKV_PLATFORM_EXTERNAL_DEPLOYMENT) ||
     p->assurances!=AKV_PLATFORM_REQUIRED_ASSURANCES || p->operation_mask!=AKV_PLATFORM_QK_PV_V1 ||
     p->qk_profile!=1 || p->pv_profile!=1 || p->nr_lanes!=4 || p->vlen_bits!=1024 ||
     !p->read_state || !p->acquire || !p->release)return 0;
  unsigned nonzero=0;
  const volatile uint8_t *identity=p->contract_sha256;
  for(size_t i=0;i<sizeof(p->contract_sha256);++i)nonzero|=identity[i];
  return nonzero!=0;
}

static akv_platform_result_t state_valid(const akv_resident_platform_provider_v1 *p,
                                       akv_platform_thread_state_v1 *s) {
  const akv_platform_result_t result=p->read_state(p->context,s);
  if(result!=AKV_PLATFORM_READY)return result==AKV_PLATFORM_BAD_VSTART?result:AKV_PLATFORM_KEEP_NATIVE;
  if(s->vstart)return AKV_PLATFORM_BAD_VSTART;
  if(s->vlenb!=128 || s->fcsr>>5 || s->vcsr>7)return AKV_PLATFORM_KEEP_NATIVE;
  return AKV_PLATFORM_READY;
}

akv_platform_result_t akv_resident_platform_admit_v1(
    const akv_resident_platform_provider_v1 *p,const akv_platform_request_v1 *r,
    akv_platform_session_v1 *out) {
  if(!provider_valid(p))return AKV_PLATFORM_KEEP_NATIVE;
  akv_platform_thread_state_v1 before,after;
  akv_platform_result_t result=state_valid(p,&before);
  if(result!=AKV_PLATFORM_READY)return result;
  /* Only now may C inspect/copy caller metadata or acquire the memory lease. */
  if(!r || !out || out->cookie || r->abi_version!=1 || r->struct_bytes!=sizeof(*r))return AKV_PLATFORM_INVALID;
  if(r->requested!=1 || r->workers!=1 || r->decode_head_rows!=1 || r->query_tokens!=1 || !r->groups || r->refill_overlap ||
     !r->spans || !r->span_count || r->span_count>32)return AKV_PLATFORM_KEEP_NATIVE;
  for(size_t i=0;i<r->span_count;++i) {
    const uintptr_t base=(uintptr_t)r->spans[i].base;
    if(!base || !r->spans[i].bytes || r->spans[i].bytes>UINTPTR_MAX-base || r->spans[i].writable>1)
      return AKV_PLATFORM_KEEP_NATIVE;
  }
  const uintptr_t cookie=p->acquire(p->context,r);
  if(!cookie)return AKV_PLATFORM_KEEP_NATIVE;
  result=state_valid(p,&after);
  if(result!=AKV_PLATFORM_READY || before.vlenb!=after.vlenb || before.fcsr!=after.fcsr || before.vcsr!=after.vcsr) {
    p->release(p->context,cookie);
    return result==AKV_PLATFORM_BAD_VSTART?result:AKV_PLATFORM_INVALID;
  }
  out->provider=p;out->cookie=cookie;out->groups=r->groups;out->prepared=0;out->started=0;
  out->spans=r->spans;out->span_count=r->span_count;
  return AKV_PLATFORM_READY;
}

static int covered(const akv_platform_session_v1 *s,uintptr_t base,size_t rows,
                   size_t stride,size_t bytes,unsigned write) {
  size_t offset,total;
  if(!base || !rows || !bytes || __builtin_mul_overflow(rows-1,stride,&offset) ||
     __builtin_add_overflow(offset,bytes,&total) || total>UINTPTR_MAX-base)return 0;
  for(size_t i=0;i<s->span_count;++i) {
    const akv_platform_span_v1 *span=&s->spans[i];
    const uintptr_t first=(uintptr_t)span->base;
    if((!write || span->writable) && base>=first && base-first<=span->bytes &&
       total<=span->bytes-(base-first))return 1;
  }
  return 0;
}

akv_platform_result_t akv_resident_platform_group_v1(
    akv_platform_session_v1 *s,unsigned group,const akv_device_t *device,
    const akv_attention_plan_t *plan,const akv_attention_features_t *features,
    akv_resident_attention_options_t *out) {
  if(!s || !s->cookie || !provider_valid(s->provider))return AKV_PLATFORM_INVALID;
  if(s->started)return AKV_PLATFORM_NO_RETRY;
  akv_platform_thread_state_v1 state;
  const akv_platform_result_t result=state_valid(s->provider,&state);
  if(result!=AKV_PLATFORM_READY)return result;
  if(!out || group!=s->prepared || group>=s->groups)return AKV_PLATFORM_INVALID;
  const akv_resident_attention_options_t options={
    .requested=1,.validated_available=1,.platform_admitted=1,.whole_call_owned=1,
    .decode_head_rows=1,.query_tokens=1,.plain_scores=1,.refill_overlap=0,
    .vlenb=state.vlenb,.frm=0,.vstart=0};
  if(akv_resident_attention_select(device,plan,&options,0)!=AKV_RESIDENT_USE_SHARED)
    return AKV_PLATFORM_KEEP_NATIVE;
  const akv_descriptor_t *d=&plan->descriptor;
  if(!covered(s,d->q_base,d->q_rows,d->q_row_stride_bytes,d->head_dim*2,0) ||
     !covered(s,d->k_base,d->kv_length,d->k_token_stride_bytes,d->head_dim*2,0) ||
     !covered(s,d->v_base,d->kv_length,d->v_token_stride_bytes,d->head_dim*2,0) ||
     !covered(s,(uintptr_t)plan->mask,1,0,d->kv_length*2,0) ||
     !covered(s,(uintptr_t)plan->output,d->q_rows,plan->output_row_stride_bytes,d->head_dim*4,1))
    return AKV_PLATFORM_KEEP_NATIVE;
  if(!akv_attention_can_use_plain_scores(features,plan->mask,plan->descriptor.kv_length))
    return AKV_PLATFORM_KEEP_NATIVE;
  *out=options;
  ++s->prepared;
  return AKV_PLATFORM_READY;
}

akv_platform_result_t akv_resident_platform_start_v1(akv_platform_session_v1 *s) {
  if(!s || !s->cookie || !provider_valid(s->provider))return AKV_PLATFORM_INVALID;
  if(s->started)return AKV_PLATFORM_NO_RETRY;
  akv_platform_thread_state_v1 state;
  const akv_platform_result_t result=state_valid(s->provider,&state);
  if(result!=AKV_PLATFORM_READY)return result;
  if(!s->groups || s->prepared!=s->groups)return AKV_PLATFORM_INVALID;
  s->started=1;return AKV_PLATFORM_READY;
}

void akv_resident_platform_finish_v1(akv_platform_session_v1 *s) {
  if(s && s->cookie) {
    s->provider->release(s->provider->context,s->cookie);
    s->cookie=0;s->provider=NULL;s->groups=0;s->prepared=0;s->started=0;s->spans=NULL;s->span_count=0;
  }
}
