#include "AckDiagnostics.hpp"
#include "CompletionEvent.hpp"
#include "LegacyCompletionEvent.hpp"
#include <cassert>
#include <fstream>
#include <chrono>
#include <cstdio>
using namespace closed_loop;
Snapshot sample(uint64_t t,unsigned steps){return Snapshot{t,t,steps?8u:7u,t,t,t,t,t-1000,true,true};}
int main(int argc,char**argv){
 assert(argc==2);unsigned checks=0;
 for(int missing=0;missing<9;++missing){
  uint64_t wall=0,sim=10000;unsigned steps=0;Snapshot last{};
  auto read=[&](Snapshot &s){++wall;s=sample(sim,steps);if(missing<8){s.available_mask&=~(uint32_t(1)<<missing);s.source_error[missing]=(missing>=3&&missing<=5)?1:2;}else{s.mode_available=false;s.mode_error=6;}last=s;return true;};
  auto result=run(10000,10,7,8,25,read,[&](){++steps;sim+=1000;return true;},[&](){return wall;},[&](){++wall;},[](){return false;});
  assert(result==Result::timeout);assert(steps==(missing==7?1u:0u));
  auto text=diagnostic(last,sim,steps?8:7,steps>0,result,25,wall,8,wall,1);
  std::ofstream(std::string(argv[1])+"/missing_"+std::to_string(missing)+".json")<<text;++checks;
 }
 for(int mode=0;mode<2;++mode){auto s=sample(10000,0);if(mode==0)s.armed=false;else s.rates_enabled=false;assert(inspect(s,10000,7,false)==State::mismatch);++checks;}
 {unsigned steps=0,calls=0;uint64_t wall=0,sim=10000;auto read=[&](Snapshot&s){++calls;++wall;s=sample(sim,steps);if(calls<3)s.received=sim-1000;return true;};
  auto result=run(10000,1,7,8,20,read,[&](){assert(calls>=3);++steps;sim+=1000;return true;},[&](){return wall;},[](){},[](){return false;});assert(result==Result::complete&&steps==1);++checks;}
 {auto s=sample(10000,0);s.received=11000;assert(inspect(s,10000,7,false)==State::mismatch);++checks;}
 {auto s=sample(10000,1);s.applied=10000;assert(inspect(s,10000,8,true)==State::mismatch);++checks;}
 {auto s=sample(10000,0);s.applied_valid=false;s.available_mask&=127;assert(inspect(s,10000,7,false)==State::ready);++checks;}
 {unsigned steps=0;uint64_t wall=0,sim=10000;auto read=[&](Snapshot&s){++wall;s=sample(sim,steps);return true;};auto result=run(10000,10,7,8,20,read,[&](){++steps;sim+=1000;return true;},[&](){return wall;},[](){},[&](){return steps==1;});assert(result==Result::cancelled&&steps==1);++checks;}
 {Snapshot s=sample(10000,0);s.gyro=9900;s.available_mask&=~8u;s.source_error[3]=1;auto c=conditions(s,10000,7,false);assert(!c[3].available&&!c[3].ready&&c[3].observed==9900&&c[3].source_error==1);++checks;}
 using Clock=std::chrono::steady_clock;
 auto now=[](){return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count());};
 Result results[2];
 for(int legacy=0;legacy<2;++legacy){
  CompletionEvent event;LegacyCompletionEvent old_event;event.notify();old_event.notify();uint64_t observed=event.generation(),old_observed=old_event.generation();unsigned calls=0,steps=0;uint64_t sim=10000;const uint64_t deadline=now()+250000;
  auto read=[&](Snapshot&s){++calls;s=sample(sim,steps);if(calls==1){s.available_mask&=~(2u|4u|64u|128u);}return true;};
  auto wait=[&](){if(legacy)old_event.wait(old_observed,Clock::time_point(std::chrono::microseconds(deadline)),[](){return false;});else event.wait(observed,Clock::time_point(std::chrono::microseconds(deadline)),[](){return false;});};
  results[legacy]=run(10000,1,7,8,deadline,read,[&](){++steps;sim+=1000;return true;},now,wait,[](){return false;});
  if(legacy)assert(results[legacy]==Result::timeout&&steps==0&&calls==1);else assert(results[legacy]==Result::complete&&steps==1&&calls>=2);++checks;
 }
 std::printf("{\"passed\":true,\"checks\":%u,\"each_condition_never_appears\":9,\"legacy_lost_requery_reproduced\":true,\"new_requery_recovers_without_new_event\":true,\"native_calls\":0}\n",checks);
}
