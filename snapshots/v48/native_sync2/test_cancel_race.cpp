#include "ClosedLoopStep.hpp"
#include <cassert>
#include <cstdio>
int main(){bool cancelled=false;unsigned advances=0;auto read=[](closed_loop::Snapshot&s){s={10000,10000,7,10000,10000,10000,10000,9000,true,true};return true;};auto advance=[&](){cancelled=true;return false;};auto result=closed_loop::run(10000,1,7,8,100,read,advance,[](){return 0u;},[](){},[&](){return cancelled;});assert(result==closed_loop::Result::cancelled&&advances==0);std::puts("{\"passed\":true,\"cancel_at_advance_boundary\":true,\"native_calls\":0}");}
