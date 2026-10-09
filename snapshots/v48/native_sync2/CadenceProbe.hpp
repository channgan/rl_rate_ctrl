#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <ctime>
namespace cadence_probe {
struct Record { uint64_t wall_ns,cpu_ns,sample,generation,a,b,sequence,event; };
static_assert(sizeof(Record)==64,"record ABI");
enum Event:uint64_t { RawPublished=1,FilterEnter,FilterExit,RawConsumed,FilteredPublished,ControlEnter,ControlExit,FilteredConsumed,InferenceEnter,InferenceExit };
inline uint64_t clock_ns(clockid_t id) noexcept {timespec t{};return clock_gettime(id,&t)==0?uint64_t(t.tv_sec)*1000000000ull+uint64_t(t.tv_nsec):0;}
template<size_t N> class alignas(4096) Ring {
 static constexpr uint64_t CLOSED=uint64_t(1)<<63;
 alignas(64) std::atomic<uint64_t> gate_{0};
 alignas(64) std::atomic<bool> started_{false};
 alignas(64) std::atomic_flag busy_=ATOMIC_FLAG_INIT;
 alignas(64) std::atomic<uint64_t> contention_{0};
 uint64_t count_=0,clock_errors_=0;
 alignas(4096) std::array<Record,N> rows_{};
 public:
 bool initialize_quiescent() noexcept {if(!gate_.is_lock_free()||!contention_.is_lock_free())return false;gate_.store(CLOSED);count_=clock_errors_=0;contention_=0;for(auto&r:rows_)r={};busy_.clear();gate_.store(0,std::memory_order_release);started_.store(true,std::memory_order_release);return true;}
 bool append(uint64_t event,uint64_t sample=0,uint64_t generation=0,uint64_t a=0,uint64_t b=0,bool cpu=false) noexcept {
  if(!started_.load(std::memory_order_acquire))return false;
  const auto state=gate_.fetch_add(1,std::memory_order_acquire);
  if(state&CLOSED){gate_.fetch_sub(1,std::memory_order_release);return false;}
  if(busy_.test_and_set(std::memory_order_acquire)){contention_.fetch_add(1,std::memory_order_relaxed);gate_.fetch_sub(1,std::memory_order_release);return false;}
  Record r{clock_ns(CLOCK_MONOTONIC),cpu?clock_ns(CLOCK_THREAD_CPUTIME_ID):0,sample,generation,a,b,count_,event};
  if(!r.wall_ns||(cpu&&!r.cpu_ns))++clock_errors_;
  rows_[count_%N]=r;++count_;busy_.clear(std::memory_order_release);gate_.fetch_sub(1,std::memory_order_release);return true;
 }
 // Begin seals admission atomically. Existing writers finish; rejected later calls never access rows/counters.
 void seal_begin() noexcept {gate_.fetch_or(CLOSED,std::memory_order_acq_rel);}
 bool writers_quiescent()const noexcept{return gate_.load(std::memory_order_acquire)==CLOSED;}
 uint64_t count_quiescent()const noexcept{return count_;}
 uint64_t contention_quiescent()const noexcept{return contention_.load();}
 uint64_t clock_errors_quiescent()const noexcept{return clock_errors_;}
 uint64_t overwritten_quiescent()const noexcept{return count_>N?count_-N:0;}
 template<class Sink> void visit_quiescent(Sink sink)const {for(uint64_t i=count_>N?count_-N:0;i<count_;++i)sink(rows_[i%N]);}
};
using TaskRing=Ring<262144>;
extern TaskRing raw,filter,control;
struct Scope {TaskRing&r;uint64_t exit;Scope(TaskRing&ring,uint64_t enter,uint64_t leave):r(ring),exit(leave){r.append(enter,0,0,0,0,true);}~Scope(){r.append(exit,0,0,0,0,true);}};
bool initialize(); // only process startup before producers
bool stop_export(); // only RL command thread; seals writers, bounded wait, then file IO
}
