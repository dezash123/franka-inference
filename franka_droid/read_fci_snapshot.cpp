#include <franka/robot.h>
#include <chrono>
#include <iomanip>
#include <iostream>
int main() {
  try {
    franka::Robot robot("172.16.0.2",franka::RealtimeConfig::kEnforce);
    auto s=robot.readOnce();
    const auto now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::cout<<std::setprecision(15)<<"{\"q\":[";
    for(size_t i=0;i<7;++i){if(i)std::cout<<',';std::cout<<s.q[i];}
    std::cout<<"],\"xyz\":["<<s.O_T_EE[12]<<','<<s.O_T_EE[13]<<','<<s.O_T_EE[14]
      <<"],\"monotonic\":"<<now<<",\"has_errors\":"<<(s.current_errors?"true":"false")
      <<",\"idle\":"<<(s.robot_mode==franka::RobotMode::kIdle?"true":"false")<<"}\n";
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
