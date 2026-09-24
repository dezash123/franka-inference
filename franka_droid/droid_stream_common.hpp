#pragma once
#include <franka/robot.h>
#include <franka/exception.h>
#include <tinyxml2.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
using Joints = std::array<double, 7>;
template<size_t N> void array(const std::array<double,N>& a) {
  std::cout << '[';
  for(size_t i=0;i<N;++i) { if(i) std::cout << ','; std::cout << a[i]; }
  std::cout << ']';
}
void stateJson(const franka::RobotState& s) {
  std::cout << "{\"q\":"; array(s.q);
  std::cout << ",\"q_d\":"; array(s.q_d);
  std::cout << ",\"dq\":"; array(s.dq);
  std::cout << ",\"xyz\":"; array(std::array<double,3>{s.O_T_EE[12],s.O_T_EE[13],s.O_T_EE[14]});
  std::cout << ",\"wrench\":"; array(s.O_F_ext_hat_K);
  std::cout << ",\"mass_kg\":" << s.m_total
            << ",\"success_rate\":" << s.control_command_success_rate
            << ",\"has_errors\":" << (s.current_errors ? "true":"false") << '}';
}
void idle(const franka::RobotState& s) {
  if(s.robot_mode!=franka::RobotMode::kIdle || s.current_errors)
    throw std::runtime_error("Robot must be idle and free of errors");
  for(double v:s.dq)
    if(!std::isfinite(v) || std::abs(v)>.02) throw std::runtime_error("Robot is not stationary");
}
}
