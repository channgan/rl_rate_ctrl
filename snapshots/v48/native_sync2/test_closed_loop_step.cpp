#include "ClosedLoopStep.hpp"
#include <cassert>
#include <cstdio>
using namespace closed_loop;
int main() {
    unsigned tests=0;
    Snapshot good{10000,10000,7,10000,10000,10000,10000,9000,true,true};
    assert(inspect(good,10000,7,true)==State::ready);++tests;
    for(int field=0;field<7;++field) {
        Snapshot s=good;
        uint64_t *fields[]={&s.clock,&s.physics,&s.gyro,&s.torque,&s.allocated,&s.received,&s.sequence};
        --*fields[field];assert(inspect(s,10000,7,true)==State::pending);++tests;
    }
    for(int field=0;field<7;++field) {
        Snapshot s=good;
        uint64_t *fields[]={&s.clock,&s.physics,&s.gyro,&s.torque,&s.allocated,&s.received,&s.sequence};
        ++*fields[field];assert(inspect(s,10000,7,true)==State::mismatch);++tests;
    }
    auto s=good;s.applied=8000;assert(inspect(s,10000,7,true)==State::mismatch);++tests;
    s=good;s.applied_valid=false;assert(inspect(s,10000,7,true)==State::mismatch);++tests;
    s=good;s.received_valid=false;assert(inspect(s,10000,7,true)==State::pending);++tests;
    uint64_t wall=0,sim=10000;unsigned steps=0,polls=0;
    auto read=[&](Snapshot &o){++wall;++polls;o={sim,sim,steps?8u:7u,sim,sim,sim,sim,sim-1000,true,true};if(polls%4)o.received=sim-1000;return true;};
    auto advance=[&](){assert(polls%4==0);++steps;sim+=1000;return true;};
    assert(run(10000,10,7,8,100,read,advance,[&](){return wall;},[](){},[](){return false;})==Result::complete);
    assert(steps==10 && polls==44);++tests;
    wall=0;steps=0;
    auto absent=[&](Snapshot &){++wall;return false;};
    assert(run(10000,10,7,8,8,absent,advance,[&](){return wall;},[](){},[](){return false;})==Result::timeout);
    assert(steps==0);++tests;
    wall=0;steps=0;sim=10000;polls=3;
    assert(run(10000,10,7,8,100,read,[](){return false;},[&](){return wall;},[](){},[](){return false;})==Result::world_step_failed);++tests;
    wall=0;steps=0;sim=10000;polls=3;
    auto jump=[&](){++steps;sim+=5000;return true;};
    assert(run(10000,10,7,8,100,read,jump,[&](){return wall;},[](){},[](){return false;})==Result::mismatch);
    assert(steps==1);++tests;
    wall=0;steps=0;sim=10000;polls=3;
    assert(run(10000,10,7,8,100,read,advance,[&](){return wall;},[](){},[](){return true;})==Result::cancelled);assert(steps==0);++tests;
    std::printf("{\"passed\":true,\"checks\":%u,\"native_calls\":0}\n",tests);
}
