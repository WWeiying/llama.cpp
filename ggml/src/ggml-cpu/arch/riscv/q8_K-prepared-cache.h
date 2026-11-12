#pragma once

#include "qbs/qbs_abi.h"
#include "rvv-byte-ops.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace ggml_riscv_q8_K_cache {

struct key {
    uint32_t k=0,producer_tag=0;
    uint8_t rows=4,profile=QBS_ACTIVATION_PROFILE_Q8_K;
    uint8_t layout=QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED,frm=0,vstart=0;
};
inline bool same(const key &a,const key &b) {
    return a.k==b.k && a.producer_tag==b.producer_tag && a.rows==b.rows &&
        a.profile==b.profile && a.layout==b.layout && a.frm==b.frm && a.vstart==b.vstart;
}
struct counters {
    uint64_t queries=0,hits=0,misses=0,stores=0,evictions=0;
    uint64_t rejected=0,allocation_failures=0,conflicts=0;
    // Requested comparison lengths, not measured memory traffic: memcmp may stop early.
    uint64_t comparison_bytes_requested=0,copied_payload_bytes=0;
};

// Independent owned snapshots. Not thread-safe: use one instance per worker.
// Retained payload <= Budget; replacement's temporary allocation <= MaxEntry.
// Caller must supply actual controls and independently captured raised FFLAGS.
template<size_t Slots=8,size_t Budget=1024*1024,size_t MaxEntry=128*1024>
class value_cache {
    static_assert(Slots>0 && Budget>0 && MaxEntry>0,"invalid cache bounds");
    struct entry {
        std::unique_ptr<uint8_t[]> data;
        key identity{};
        size_t input_bytes=0,output_bytes=0;
        unsigned flags=0;
        uint64_t age=0;
        bool valid=false;
    };
    std::array<entry,Slots> entries_{};
    size_t retained_=0;
    uint64_t clock_=0;
    counters stats_{};
    static uint8_t *allocate_default(size_t n) {return new(std::nothrow) uint8_t[n];}
public:
    using allocator=uint8_t *(*)(size_t);
private:
    allocator allocate_;
    static bool overlap(const void *a,size_t na,const void *b,size_t nb) {
        const uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
        return x>=y ? x-y<nb:y-x<na;
    }
    static bool valid(const key &k,size_t in,size_t out) {
        if(!k.k || k.k%QBS_Q8_K_BLOCK_ELEMENTS || !k.producer_tag || k.rows!=4 ||
           k.profile!=QBS_ACTIVATION_PROFILE_Q8_K || k.layout!=QBS_ACTIVATION_LAYOUT_M4_INTERLEAVED ||
           k.frm || k.vstart || k.k>std::numeric_limits<size_t>::max()/16) return false;
        size_t blocks=k.k/QBS_Q8_K_BLOCK_ELEMENTS;
        if(blocks>std::numeric_limits<size_t>::max()/(4*QBS_Q8_K_BLOCK_BYTES)) return false;
        return in==size_t(k.k)*16 && out==blocks*4*QBS_Q8_K_BLOCK_BYTES &&
            in<=Budget && out<=Budget-in && in<=MaxEntry && out<=MaxEntry-in;
    }
    void erase(size_t i) {
        if(entries_[i].valid) retained_-=entries_[i].input_bytes+entries_[i].output_bytes;
        entries_[i].valid=false;entries_[i].data.reset();
    }
    uint64_t tick() {
        if(clock_==std::numeric_limits<uint64_t>::max()) clear();
        return ++clock_;
    }
    size_t find(const key &k,const void *input,size_t in,size_t out) {
        for(size_t i=0;i<Slots;++i) {
            const auto &e=entries_[i];
            if(!e.valid || !same(e.identity,k) || e.input_bytes!=in || e.output_bytes!=out) continue;
            stats_.comparison_bytes_requested+=in;
            if(ggml_riscv_byte_ops::equal(e.data.get(),input,in)) return i;
        }
        return Slots;
    }
public:
    explicit value_cache(allocator allocate=&allocate_default):allocate_(allocate) {}
    value_cache(const value_cache &)=delete;
    value_cache &operator=(const value_cache &)=delete;
    const counters &statistics() const {return stats_;}
    size_t retained_bytes() const {return retained_;}
    size_t size() const {
        size_t n=0;for(const auto &e:entries_) n+=e.valid;return n;
    }
    void clear() {
        for(size_t i=0;i<Slots;++i) erase(i);
        clock_=0;
    }
    bool get(const key &k,const void *input,size_t in,void *output,size_t out,unsigned &flags) {
        ++stats_.queries;
        if(!input || !output || !valid(k,in,out) || overlap(input,in,output,out)) {
            ++stats_.rejected;++stats_.misses;return false;
        }
        const uint64_t age=tick();
        size_t i=find(k,input,in,out);
        if(i==Slots) {++stats_.misses;return false;}
        auto &e=entries_[i];
        ggml_riscv_byte_ops::copy(output,e.data.get()+in,out);flags=e.flags;e.age=age;
        ++stats_.hits;stats_.copied_payload_bytes+=out;return true;
    }
    bool put(const key &k,const void *input,size_t in,const void *output,size_t out,unsigned flags) {
        if(!input || !output || flags>31 || !valid(k,in,out) || overlap(input,in,output,out)) {
            ++stats_.rejected;return false;
        }
        const uint64_t age=tick();
        size_t duplicate=find(k,input,in,out);
        if(duplicate!=Slots) {
            auto &e=entries_[duplicate];
            if(e.flags!=flags || !ggml_riscv_byte_ops::equal(e.data.get()+in,output,out)) {
                ++stats_.conflicts;erase(duplicate);return false;
            }
            e.age=age;return true;
        }
        const size_t total=in+out;
        std::unique_ptr<uint8_t[]> fresh(allocate_ ? allocate_(total):nullptr);
        if(!fresh) {++stats_.allocation_failures;return false;}
        // Populate before eviction: allocation failure never destroys a useful entry.
        ggml_riscv_byte_ops::copy(fresh.get(),input,in);ggml_riscv_byte_ops::copy(fresh.get()+in,output,out);
        while(retained_>Budget-total || size()==Slots) {
            size_t victim=Slots;
            for(size_t i=0;i<Slots;++i)
                if(entries_[i].valid && (victim==Slots || entries_[i].age<entries_[victim].age)) victim=i;
            erase(victim);++stats_.evictions;
        }
        size_t slot=0;while(entries_[slot].valid) ++slot;
        auto &e=entries_[slot];e.data=std::move(fresh);e.identity=k;
        e.input_bytes=in;e.output_bytes=out;e.flags=flags;e.age=age;e.valid=true;
        retained_+=total;++stats_.stores;stats_.copied_payload_bytes+=total;return true;
    }
};
} // namespace ggml_riscv_q8_K_cache
