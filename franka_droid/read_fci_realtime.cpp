#include "franka_realtime.hpp"
#include <franka/robot.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <algorithm>
#include <array>
#include <cmath>

// Live state reception only: no control(), load changes, recovery, or actuator writes.
int main() {
  try {
    franka_rt::configureIo();
    franka::Robot robot("172.16.0.2",franka::RealtimeConfig::kEnforce);
    franka_rt::configureIo();franka_rt::lockMemory();
    std::cout<<std::setprecision(12)<<std::unitbuf;
    std::cout<<"{\"io_thread\":"<<franka_rt::threadJson()<<",\"realtime_config\":"<<franka_rt::configJson()<<"}\n";
    std::string error;
    std::thread receiver([&] {
      try {
        franka_rt::configureControl();
        std::cout<<"{\"receive_thread\":"<<franka_rt::threadJson()<<"}\n";
        using Clock=std::chrono::steady_clock;
        auto begin=Clock::now(),previous=begin;
        double minimum_ms=1e9,maximum_ms=0,total_ms=0,robot_max_gap_ms=0;
        uint64_t count=0,over_2ms=0,over_10ms=0,previous_robot_ms=0,robot_skipped_ticks=0;
        int wrong_cpu=0;double max_velocity=0;bool any_errors=false;
        const int expected_cpu=franka_rt::number("FRANKA_RT_CPU");
        robot.read([&](const franka::RobotState& state) {
          auto now=Clock::now();double dt=std::chrono::duration<double,std::milli>(now-previous).count();
          if(count) {minimum_ms=std::min(minimum_ms,dt);maximum_ms=std::max(maximum_ms,dt);total_ms+=dt;
            if(dt>2)++over_2ms;
            if(dt>10)++over_10ms;
            auto gap=state.time.toMSec()-previous_robot_ms;robot_max_gap_ms=std::max(robot_max_gap_ms,double(gap));
            if(gap>1)robot_skipped_ticks+=gap-1;
          }
          if(sched_getcpu()!=expected_cpu)++wrong_cpu;
          for(double velocity:state.dq)max_velocity=std::max(max_velocity,std::abs(velocity));
          any_errors=any_errors||bool(state.current_errors);
          previous=now;previous_robot_ms=state.time.toMSec();++count;
          return std::chrono::duration<double>(now-begin).count()<10;
        });
        double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
        std::cout<<"{\"read_only\":true,\"elapsed_s\":"<<elapsed<<",\"state_count\":"<<count
          <<",\"host_interval_min_ms\":"<<minimum_ms<<",\"host_interval_mean_ms\":"<<total_ms/double(count-1)
          <<",\"host_interval_max_ms\":"<<maximum_ms<<",\"host_gaps_over_2ms\":"<<over_2ms
          <<",\"host_gaps_over_10ms\":"<<over_10ms<<",\"robot_max_gap_ms\":"<<robot_max_gap_ms
          <<",\"robot_skipped_ticks\":"<<robot_skipped_ticks<<",\"wrong_cpu_samples\":"<<wrong_cpu
          <<",\"maximum_measured_joint_speed_rad_s\":"<<max_velocity
          <<",\"robot_reported_errors\":"<<(any_errors?"true":"false")<<",\"motion_commands_sent\":0}\n";
      } catch(const std::exception& e) {error=e.what();}
    });
    receiver.join();if(!error.empty())throw std::runtime_error(error);
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
