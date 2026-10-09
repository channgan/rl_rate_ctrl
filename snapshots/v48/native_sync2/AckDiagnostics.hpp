#pragma once
#include "ClosedLoopStep.hpp"
#include <array>
#include <sstream>
namespace closed_loop {
struct Condition {
    const char *name;
    bool required, available, ready;
    uint64_t expected, observed, expected_step, observed_step, generation;
    bool observed_step_known;
    uint32_t source_error;
};
inline std::array<Condition,10> conditions(const Snapshot &s,uint64_t t,uint64_t seq,bool after_step){
    const uint64_t observed[]={s.clock,s.physics,s.sequence,s.gyro,s.torque,s.allocated,s.received,s.applied};
    const uint64_t expected[]={t,t,seq,t,t,t,t,t>=1000?t-1000:0};
    const char *names[]={"clock","physics","step_sequence","filter","torque","allocator","GZ_receive","actual_apply"};
    std::array<Condition,10> result{};
    for(unsigned i=0;i<8;++i){
        bool required=i!=7 || after_step;
        bool available=(s.available_mask&(uint32_t(1)<<i))!=0;
        bool flag=i==6?s.received_valid:(i==7?s.applied_valid:true);
        result[i]={names[i],required,available,!required||(available&&flag&&observed[i]==expected[i]),expected[i],observed[i],seq,s.sequence,
            (i>=3&&i<=5)?s.uorb_generation[i-3]:0,i==1||i==2||i==6||i==7,s.source_error[i]};
    }
    result[8]={"armed",true,s.mode_available,s.mode_available&&s.armed,1,uint64_t(s.armed),seq,0,0,false,s.mode_error};
    result[9]={"rates_enabled",true,s.mode_available,s.mode_available&&s.rates_enabled,1,uint64_t(s.rates_enabled),seq,0,0,false,s.mode_error};
    return result;
}
inline const char *reason(Result r){switch(r){case Result::complete:return "complete";case Result::timeout:return "timeout";case Result::mismatch:return "mismatch";case Result::world_step_failed:return "world_step_failed";case Result::cancelled:return "cancelled";case Result::capture_failed:return "capture_failed";}return "unknown";}
inline std::string escaped(const std::string &text){std::string out;for(char c:text){if(c=='"'||c=='\\'){out+='\\';out+=c;}else if(static_cast<unsigned char>(c)<32){out+='?';}else out+=c;}return out;}
inline std::string diagnostic(const Snapshot &s,uint64_t target,uint64_t expected_seq,bool after_step,
    Result result,uint64_t deadline,uint64_t now,uint64_t requested_seq,uint64_t request_calls,uint64_t request_errors){
    std::ostringstream o;o<<"{\"schema\":1,\"outcome\":\""<<reason(result)<<"\",\"cancelled\":"<<(result==Result::cancelled?"true":"false")
      <<",\"timeout\":"<<(result==Result::timeout?"true":"false")<<",\"expected_sample_us\":"<<target<<",\"requested_step_seq\":"<<requested_seq
      <<",\"phase\":\""<<(after_step?"after_integrate":"before_first_integrate")<<"\",\"wall_us\":"<<now<<",\"deadline_wall_us\":"<<deadline
      <<",\"ack_request_calls\":"<<request_calls<<",\"ack_request_errors\":"<<request_errors
      <<",\"ack_transport_elapsed_us\":"<<s.rpc_elapsed_us<<",\"ack_response_size\":"<<s.ack_response_size<<",\"source_error_detail\":\""<<escaped(s.rpc_error)<<"\""
      <<",\"source_error_codes\":{\"0\":\"none\",\"1\":\"uORB_copy_unavailable\",\"2\":\"transport_request_failed\",\"3\":\"service_rejected\",\"4\":\"ACK_schema_invalid\",\"5\":\"ACK_value_invalid\",\"6\":\"mode_copy_unavailable\",\"7\":\"request_exception\"},\"conditions\":[";
    bool first=true;for(const auto &c:conditions(s,target,expected_seq,after_step)){
        if(!first){o<<',';}
        first=false;
        o<<"{\"name\":\""<<c.name<<"\",\"required\":"<<(c.required?"true":"false")<<",\"available\":"<<(c.available?"true":"false")
         <<",\"ready\":"<<(c.ready?"true":"false")<<",\"expected_timestamp_or_seq\":"<<c.expected<<",\"last_observed_timestamp_or_seq\":"<<c.observed
         <<",\"expected_step_seq\":"<<c.expected_step<<",\"observed_step_seq\":";
        if(c.observed_step_known)o<<c.observed_step;else o<<"null";
        o<<",\"observed_uorb_generation\":"<<c.generation<<",\"source_error\":"<<c.source_error<<'}';
    }o<<"]}";return o.str();
}
}
