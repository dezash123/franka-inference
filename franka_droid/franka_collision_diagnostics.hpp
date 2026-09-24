#pragma once

#include <franka/robot_state.h>
#include <iostream>

// Serialize on the I/O thread, or after control has stopped; never in the
// real-time callback. Preserve both wrench frames and native collision flags.
inline void collisionDiagnosticsJson(const franka::RobotState& s) {
  std::cout << "{\"robot_time_ms\":" << s.time.toMSec()
            << ",\"robot_mode\":" << s.robot_mode
            << ",\"current_errors\":" << s.current_errors
            << ",\"last_motion_errors\":" << s.last_motion_errors;
  auto field = [](const char* name, const auto& values) {
    std::cout << ",\"" << name << "\":[";
    bool first = true;
    for (double value : values) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << value;
    }
    std::cout << ']';
  };
  field("q", s.q);
  field("dq", s.dq);
  field("q_d", s.q_d);
  field("dq_d", s.dq_d);
  field("ddq_d", s.ddq_d);
  field("wrench_base", s.O_F_ext_hat_K);
  field("wrench_stiffness", s.K_F_ext_hat_K);
  field("external_joint_torque", s.tau_ext_hat_filtered);
  field("measured_joint_torque", s.tau_J);
  field("commanded_joint_torque", s.tau_J_d);
  field("joint_contact", s.joint_contact);
  field("joint_collision", s.joint_collision);
  field("cartesian_contact", s.cartesian_contact);
  field("cartesian_collision", s.cartesian_collision);
  field("O_T_EE", s.O_T_EE);
  field("F_T_EE", s.F_T_EE);
  field("EE_T_K", s.EE_T_K);
  field("F_x_Ctotal", s.F_x_Ctotal);
  field("I_total", s.I_total);
  std::cout << ",\"mass_ee_kg\":" << s.m_ee
            << ",\"mass_load_kg\":" << s.m_load
            << ",\"mass_total_kg\":" << s.m_total
            << ",\"control_command_success_rate\":" << s.control_command_success_rate
            << '}';
}
