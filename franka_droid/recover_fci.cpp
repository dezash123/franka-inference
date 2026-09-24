#include <franka/robot.h>
#include <franka/exception.h>
#include <iostream>
#include <string>
int main(int argc,char** argv) {
  if(argc!=2 || std::string(argv[1])!="--recover") return 2;
  try {
    franka::Robot robot("172.16.0.2",franka::RealtimeConfig::kEnforce);
    const auto before=robot.readOnce();
    std::cout<<"Before recovery: "<<before.current_errors<<std::endl;
    robot.automaticErrorRecovery();
    const auto after=robot.readOnce();
    std::cout<<"After recovery: "<<after.current_errors<<std::endl;
    return after.current_errors?1:0;
  } catch(const franka::Exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
