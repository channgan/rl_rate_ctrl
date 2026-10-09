#include "CompletionEvent.hpp"
#include <cassert>
#include <thread>
#include <atomic>
#include <cstdio>
int main(){
 using C=std::chrono::steady_clock;closed_loop::CompletionEvent e;unsigned tests=0;
 auto old=e.generation();e.notify();auto start=C::now();e.wait(old,start+std::chrono::seconds(1),[](){return false;});assert(C::now()-start<std::chrono::milliseconds(100));++tests;
 old=e.generation();std::thread producer([&](){std::this_thread::sleep_for(std::chrono::milliseconds(5));e.notify();});e.wait(old,C::now()+std::chrono::seconds(1),[](){return false;});producer.join();assert(e.generation()>old);++tests;
 old=e.generation();start=C::now();e.wait(old,start+std::chrono::milliseconds(10),[](){return false;});assert(C::now()-start>=std::chrono::milliseconds(10));++tests;
 old=e.generation();start=C::now();e.wait(old,start+std::chrono::seconds(1),[](){return true;});assert(C::now()-start<std::chrono::milliseconds(100));++tests;
 old=e.generation();e.notify();e.notify();e.wait(old,C::now()+std::chrono::seconds(1),[](){return false;});assert(e.generation()==old+2);++tests;
 std::printf("{\"passed\":true,\"event_checks\":%u,\"native_calls\":0}\n",tests);
}
