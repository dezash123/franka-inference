#include <franka/robot.h>
#include <iostream>
int main(){try{franka::Robot robot("172.16.0.2");auto s=robot.readOnce();std::cout<<"mode="<<s.robot_mode<<"\ncurrent="<<s.current_errors<<"\nlast_motion="<<s.last_motion_errors<<"\nq=";for(auto v:s.q)std::cout<<v<<",";std::cout<<"\nwrench=";for(auto v:s.O_F_ext_hat_K)std::cout<<v<<",";std::cout<<"\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
