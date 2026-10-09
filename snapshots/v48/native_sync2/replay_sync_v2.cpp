#include "ClosedLoopStep.hpp"
#include <array>
#include <fstream>
#include <cassert>
#include <cstdio>
int main(int argc,char**argv){
 assert(argc==2);std::ifstream in(argv[1],std::ios::binary);char magic[8];in.read(magic,8);assert(std::string(magic,8)=="SYNC0001");
 std::array<uint64_t,12> row{},previous{};uint64_t current_request=0,target=0,first_time=0,rows=0,transitions=0;unsigned in_group=0;
 while(in.read(reinterpret_cast<char*>(row.data()),96)){
  if(row[0]==0)break;
  if(!rows){current_request=row[10];target=row[0];first_time=target;in_group=0;}
  else if(row[10]!=current_request){assert(row[10]==current_request+1);current_request=row[10];target=previous[0];in_group=0;}
  else{target+=1000;++in_group;++transitions;}
  closed_loop::Snapshot s{row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],bool(row[8]),bool(row[9])};
  assert(closed_loop::inspect(s,target,in_group?current_request:current_request-1,in_group>0)==closed_loop::State::ready);
  previous=row;++rows;
 }
 assert(rows && row[0]==0 && row[10]==previous[10]+1);
 closed_loop::Snapshot prior{previous[0],previous[1],previous[2],previous[3],previous[4],previous[5],previous[6],previous[7],bool(previous[8]),bool(previous[9])};
 assert(closed_loop::inspect(prior,previous[0],row[10]-1,false)==closed_loop::State::ready);
 std::printf("{\"passed\":true,\"timestamp_rows\":%llu,\"one_ms_integrations\":%llu,\"first_sim_us\":%llu,\"last_sim_us\":%llu,\"next_precondition_from_last_good_snapshot\":true,\"arming_and_source_availability_not_recorded_in_old_trace\":true,\"native_calls\":0}\n",
 (unsigned long long)rows,(unsigned long long)transitions,(unsigned long long)first_time,(unsigned long long)previous[0]);
}
