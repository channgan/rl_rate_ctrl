#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
namespace closed_loop {
class CompletionEvent {
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t count{0};
public:
    void notify() { {std::lock_guard<std::mutex> lock(mutex);++count;} changed.notify_all(); }
    uint64_t generation() {std::lock_guard<std::mutex> lock(mutex);return count;}
    template<class Cancel>
    void wait(uint64_t observed, std::chrono::steady_clock::time_point deadline, Cancel cancel) {
        std::unique_lock<std::mutex> lock(mutex);
        if(count==observed && !cancel()) {
            const auto now=std::chrono::steady_clock::now();if(now>=deadline)return;
            // Event-driven progress. The bounded wake also checks the existing
            // explicit stop file; it never permits a simulator step by itself.
            changed.wait_until(lock,std::min(deadline,now+std::chrono::milliseconds(100)),[&](){return count!=observed;});
        }
    }
};
}
