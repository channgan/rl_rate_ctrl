#pragma once
#include <cstdint>
#include <array>
#include <string>
namespace closed_loop {
struct Snapshot {
    uint64_t clock{}, physics{}, sequence{}, gyro{}, torque{}, allocated{}, received{}, applied{};
    bool received_valid{}, applied_valid{};
    uint32_t available_mask{255};std::array<uint32_t,8> source_error{};std::array<uint64_t,3> uorb_generation{};
    bool mode_available{true},armed{true},rates_enabled{true};uint32_t mode_error{0};
    uint64_t rpc_elapsed_us{0};int ack_response_size{-1};std::string rpc_error{};
};
enum class State { pending, ready, mismatch };
enum class Result { complete, timeout, mismatch, world_step_failed, cancelled, capture_failed };
inline State inspect(const Snapshot &s, uint64_t t, uint64_t seq, bool check_applied) {
    if(s.mode_available&&(!s.armed||!s.rates_enabled))return State::mismatch;
    const uint32_t required=check_applied?255:127;
    if((s.available_mask&required)!=required||!s.mode_available)return State::pending;
    if (s.clock > t || s.physics > t || s.sequence > seq || s.gyro > t ||
        s.torque > t || s.allocated > t || s.received > t) return State::mismatch;
    if (s.clock != t || s.physics != t || s.sequence != seq || s.gyro != t ||
        s.torque != t || s.allocated != t || !s.received_valid || s.received != t)
        return State::pending;
    // PreUpdate for the just-finished interval must have consumed the preceding
    // boundary's motor command. Equality, not >= or wall-time delay.
    if (check_applied && (!s.applied_valid || s.applied != t-1000)) return State::mismatch;
    return State::ready;
}
template<class Read, class Advance, class Now, class Idle, class Cancel>
Result run(uint64_t start, unsigned ticks, uint64_t previous_seq, uint64_t next_seq,
           uint64_t deadline, Read read, Advance advance, Now now, Idle idle, Cancel cancel) {
    for (unsigned i=0; i<=ticks; ++i) {
        const uint64_t target=start+uint64_t(i)*1000;
        for (;;) {
            if (cancel()) return Result::cancelled;
            if (now() >= deadline) return Result::timeout;
            Snapshot s{};
            const bool available=read(s);
            if (cancel()) return Result::cancelled;
            if (now() >= deadline) return Result::timeout;
            if (available) {
                const auto state=inspect(s,target,i ? next_seq : previous_seq,i!=0);
                if (state==State::mismatch) return Result::mismatch;
                if (state==State::ready) break;
            }
            idle();
        }
        if (cancel()) return Result::cancelled;
        if (i<ticks && !advance()) {return cancel()?Result::cancelled:Result::world_step_failed;}
    }
    return Result::complete;
}
}
