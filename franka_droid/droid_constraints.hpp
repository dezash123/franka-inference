#pragma once
#include <franka/model.h>
#include <franka/robot_state.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Constraint values and safety feedback from the pinned DROID FR3 configuration
// and fairo client. See config/reference/sources.json for provenance.
namespace droid {
constexpr std::array<double,7> lower{-2.65,-1.68,-2.8,-2.95,-2.7,0.45,-2.9};
constexpr std::array<double,7> upper{2.65,1.68,2.8,-0.16,2.7,4.4,2.9};
constexpr std::array<double,7> velocity{2.075,2.075,2.075,2.075,2.51,2.51,2.51};
constexpr std::array<double,7> torque_limit{86.0,86.0,86.0,86.0,11.5,11.5,11.5};
constexpr std::array<double,3> xyz_lower{-1.0,-1.0,-1.0};
constexpr std::array<double,3> xyz_upper{1.0,1.0,1.0};
constexpr std::array<double,7> Kq{40.0,30.0,50.0,25.0,35.0,25.0,10.0};
constexpr std::array<double,7> Kqd{4.0,6.0,5.0,5.0,3.0,2.0,1.0};
constexpr std::array<double,6> Kx{400.0,400.0,400.0,15.0,15.0,15.0};
constexpr std::array<double,6> Kxd{37.0,37.0,37.0,2.0,2.0,2.0};
constexpr const char* profile_json=R"PROFILE({"sources":{"droid_commit":"33ae6a67274f36d2e29525b86f23a56616ef43a7","fairo_commit":"0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2","sources":{"franka_hardware.yaml":"https://raw.githubusercontent.com/droid-dataset/droid/33ae6a67274f36d2e29525b86f23a56616ef43a7/config/fr3/franka_hardware.yaml","franka_panda_client.cpp":"https://raw.githubusercontent.com/facebookresearch/fairo/0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2/polymetis/polymetis/src/clients/franka_panda_client/franka_panda_client.cpp","robot_ik_solver.py":"https://raw.githubusercontent.com/droid-dataset/droid/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/robot_ik/robot_ik_solver.py"}},"limits":{"cartesian_pos_upper":[1.0,1.0,1.0],"cartesian_pos_lower":[-1.0,-1.0,-1.0],"joint_pos_upper":[2.65,1.68,2.8,-0.16,2.7,4.4,2.9],"joint_pos_lower":[-2.65,-1.68,-2.8,-2.95,-2.7,0.45,-2.9],"joint_vel":[2.075,2.075,2.075,2.075,2.51,2.51,2.51],"elbow_vel":2.075,"joint_torques":[86.0,86.0,86.0,86.0,11.5,11.5,11.5]},"collision_behavior":{"lower_torque":[40.0,40.0,40.0,40.0,40.0,40.0,40.0],"upper_torque":[40.0,40.0,40.0,40.0,40.0,40.0,40.0],"lower_force":[40.0,40.0,40.0,40.0,40.0,40.0],"upper_force":[40.0,40.0,40.0,40.0,40.0,40.0]},"safety_controller":{"is_active":true,"margins":{"cartesian_pos":0.05,"joint_pos":0.2,"joint_vel":0.5},"stiffness":{"cartesian_pos":200.0,"joint_pos":50.0,"joint_vel":20.0}},"limit_rate":true,"lpf_cutoff_frequency":100,"additional_ruckig_caps":null,"normalized_action_max_joint_increment_rad":0.2,"semantics":{"hard_state_limits":"Throw on measured workspace, joint position, joint velocity or elbow velocity violations.","soft_feedback":"Add restoring Cartesian/Jacobian and joint torques within each margin; this is not target clipping.","commanded_torques":"Clamp policy torque plus safety torque per joint before the native rate limiter and filter.","collision_thresholds":"Apply all four arrays via native setCollisionBehavior before control.","controller":"External hybrid joint/Cartesian torque feedback using live FR3 model, zero desired velocity and Coriolis compensation. No Ruckig motion generator."},"controller_gains":{"Kq":[40,30,50,25,35,25,10],"Kqd":[4,6,5,5,3,2,1],"Kx":[400,400,400,15,15,15],"Kxd":[37,37,37,2,2,2]},"implementation_differences":["Live FR3 firmware model instead of the published legacy Panda URDF model","Local C++ torque callback instead of Polymetis gRPC","No automatic recovery or restart after controller errors","Target paths outside configured bounds are held before dispatch rather than waiting for a measured hard limit violation","MoveIt self-collision checks, camera freshness, command watchdog and pre-dispatch target workspace checks retained"],"profile_name":"droid_fr3"})PROFILE";
constexpr double filter_hz=100.;
constexpr double joint_margin=.2, velocity_margin=.5, cartesian_margin=.05;
constexpr double joint_stiffness=50., velocity_stiffness=20., cartesian_stiffness=200.;
constexpr double elbow_velocity_limit=2.075;
inline double reflex(double value,double lo,double hi,double margin,double k) {
  if(!std::isfinite(value) || value<lo || value>hi)
    throw std::runtime_error("DROID hard state limit exceeded");
  if(value>hi-margin) return -k*(value-hi+margin);
  if(value<lo+margin) return k*(lo+margin-value);
  return 0.;
}
inline void stateLimits(const franka::RobotState& s) {
  if(s.current_errors) throw std::runtime_error("Robot reports active control errors");
  for(size_t j=0;j<7;++j) {
    if(!std::isfinite(s.q[j]) || s.q[j]<lower[j] || s.q[j]>upper[j])
      throw std::runtime_error("DROID joint position limit at joint "+std::to_string(j+1));
    if(!std::isfinite(s.dq[j]) || std::abs(s.dq[j])>velocity[j])
      throw std::runtime_error("DROID joint speed limit at joint "+std::to_string(j+1));
  }
  for(size_t axis=0;axis<3;++axis)
    if(!std::isfinite(s.O_T_EE[12+axis]) || s.O_T_EE[12+axis]<xyz_lower[axis] || s.O_T_EE[12+axis]>xyz_upper[axis])
      throw std::runtime_error("DROID Cartesian workspace hard limit");
  if(!std::isfinite(s.delbow_c[0]) || std::abs(s.delbow_c[0])>elbow_velocity_limit)
    throw std::runtime_error("DROID elbow velocity limit");
}
inline bool targetPath(const franka::RobotState& s,const std::array<double,7>& target,const franka::Model& model) {
  double delta=0.;for(size_t j=0;j<7;++j) delta=std::max(delta,std::abs(target[j]-s.q[j]));
  const int count=std::max(4,static_cast<int>(std::ceil(delta/.005)));
  for(int step=0;step<=count;++step) {
    std::array<double,7> q{};
    for(size_t j=0;j<7;++j) q[j]=s.q[j]+(target[j]-s.q[j])*static_cast<double>(step)/count;
    auto pose=model.pose(franka::Frame::kEndEffector,q,s.F_T_EE,s.EE_T_K);
    for(size_t axis=0;axis<3;++axis)
      if(!std::isfinite(pose[12+axis]) || pose[12+axis]<xyz_lower[axis] || pose[12+axis]>xyz_upper[axis]) return false;
  }
  return true;
}
struct TorqueResult {std::array<double,7> torque{};unsigned safety_mask=0;unsigned saturation_mask=0;};
inline TorqueResult torques(const franka::RobotState& s,const std::array<double,7>& target,
                            const std::array<double,42>& J,const std::array<double,7>& coriolis) {
  // DROID HybridJointSpacePD: (Kq + J'KxJ)*position_error - (Kqd + J'KxdJ)*dq.
  // Franka already compensates gravity. Clamp only after adding safety torque.
  std::array<double,6> error{},velocity_error{},wrench{};
  std::array<double,3> safety_force{};
  TorqueResult out;
  for(size_t axis=0;axis<6;++axis) {
    for(size_t j=0;j<7;++j) {
      error[axis]+=J[axis+6*j]*(target[j]-s.q[j]);
      velocity_error[axis]-=J[axis+6*j]*s.dq[j];
    }
    wrench[axis]=Kx[axis]*error[axis]+Kxd[axis]*velocity_error[axis];
  }
  for(size_t axis=0;axis<3;++axis) {
    safety_force[axis]=reflex(s.O_T_EE[12+axis],xyz_lower[axis],xyz_upper[axis],cartesian_margin,cartesian_stiffness);
    if(safety_force[axis]!=0.) out.safety_mask|=1u<<axis;
  }
  for(size_t j=0;j<7;++j) {
    double tau=Kq[j]*(target[j]-s.q[j])-Kqd[j]*s.dq[j]+coriolis[j];
    for(size_t axis=0;axis<6;++axis) tau+=J[axis+6*j]*wrench[axis];
    for(size_t axis=0;axis<3;++axis) tau+=J[axis+6*j]*safety_force[axis];
    const double q_safety=reflex(s.q[j],lower[j],upper[j],joint_margin,joint_stiffness);
    const double v_safety=reflex(s.dq[j],-velocity[j],velocity[j],velocity_margin,velocity_stiffness);
    if(q_safety!=0.)out.safety_mask|=1u<<(3+j);
    if(v_safety!=0.)out.safety_mask|=1u<<(10+j);
    tau+=q_safety+v_safety;
    if(!std::isfinite(tau))throw std::runtime_error("Nonfinite DROID torque");
    if(std::abs(tau)>torque_limit[j])out.saturation_mask|=1u<<j;
    out.torque[j]=std::clamp(tau,-torque_limit[j],torque_limit[j]);
  }
  return out;
}
} // namespace droid
