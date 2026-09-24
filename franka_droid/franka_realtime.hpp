#pragma once
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace franka_rt {
inline std::string setting(const char* key) {
  const char* value=std::getenv(key);
  if(!value || !*value) throw std::runtime_error(std::string("Missing realtime setting: ")+key);
  return value;
}
inline int number(const char* key) {
  std::string value=setting(key);size_t used=0;int result=std::stoi(value,&used);
  if(used!=value.size()) throw std::runtime_error("Invalid realtime numeric setting");
  return result;
}
inline cpu_set_t cpuSet(const std::string& list) {
  cpu_set_t mask;CPU_ZERO(&mask);std::istringstream input(list);std::string part;
  while(std::getline(input,part,',')) {
    auto dash=part.find('-');int first=std::stoi(part),last=dash==std::string::npos?first:std::stoi(part.substr(dash+1));
    if(first<0 || last<first || last>=CPU_SETSIZE) throw std::runtime_error("Invalid realtime CPU set");
    for(int cpu=first;cpu<=last;++cpu) CPU_SET(cpu,&mask);
  }
  if(CPU_COUNT(&mask)==0) throw std::runtime_error("Empty realtime CPU set");
  return mask;
}
inline void requireSlice() {
  std::ifstream input("/proc/self/cgroup");std::string line;bool found=false;
  while(std::getline(input,line)) if(line.rfind("0::/franka.slice/",0)==0) found=true;
  if(!found) throw std::runtime_error("Launch through run-policy-stream.sh in franka.slice");
}
inline void affinity(const cpu_set_t& mask) {
  int error=pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask);
  if(error) throw std::runtime_error(std::string("CPU affinity failed: ")+std::strerror(error));
}
inline void policy(int policy,int priority) {
  sched_param param{};param.sched_priority=priority;
  int error=pthread_setschedparam(pthread_self(),policy,&param);
  if(error) throw std::runtime_error(std::string("Scheduling failed: ")+std::strerror(error));
}
inline void configureIo() {
  requireSlice();policy(SCHED_OTHER,0);
  auto mask=cpuSet(setting("FRANKA_IO_CPUS"));
  int cpu=number("FRANKA_RT_CPU");
  if(cpu<0 || cpu>=CPU_SETSIZE || CPU_ISSET(cpu,&mask)) throw std::runtime_error("Control and I/O CPUs overlap");
  affinity(mask);pthread_setname_np(pthread_self(),"franka-io");
}
inline void lockMemory() {
  if(mlockall(MCL_CURRENT|MCL_FUTURE)!=0)
    throw std::runtime_error(std::string("Realtime memory lock failed: ")+std::strerror(errno));
}
inline void configureControl() {
  requireSlice();int cpu=number("FRANKA_RT_CPU"),priority=number("FRANKA_RT_PRIORITY");
  if(cpu<0 || cpu>=CPU_SETSIZE || priority<1 || priority>98) throw std::runtime_error("Invalid control scheduling settings");
  cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);affinity(mask);policy(SCHED_FIFO,priority);
  pthread_setname_np(pthread_self(),"franka-control");
  volatile char prefault[65536];
  for(size_t i=0;i<sizeof(prefault);i+=4096) prefault[i]=0;
}
inline std::string threadJson() {
  cpu_set_t mask;CPU_ZERO(&mask);int error=pthread_getaffinity_np(pthread_self(),sizeof(mask),&mask);
  if(error) throw std::runtime_error("Cannot read CPU affinity");
  int scheduling=0;sched_param param{};
  if(pthread_getschedparam(pthread_self(),&scheduling,&param)) throw std::runtime_error("Cannot read thread priority");
  std::ostringstream out;out<<"{\"tid\":"<<syscall(SYS_gettid)<<",\"cpu\":"<<sched_getcpu()
    <<",\"scheduler\":\""<<(scheduling==SCHED_FIFO?"SCHED_FIFO":scheduling==SCHED_OTHER?"SCHED_OTHER":"other")
    <<"\",\"priority\":"<<param.sched_priority<<",\"allowed_cpus\":[";
  bool first=true;for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&mask)) {if(!first)out<<',';first=false;out<<cpu;}
  out<<"]}";return out.str();
}
inline std::string configJson() {
  std::ostringstream out;out<<"{\"control_cpu\":"<<number("FRANKA_RT_CPU")
    <<",\"control_fifo_priority\":"<<number("FRANKA_RT_PRIORITY")
    <<",\"io_cpu_list\":\""<<setting("FRANKA_IO_CPUS")<<"\",\"memory_locked\":true}";
  return out.str();
}
}
