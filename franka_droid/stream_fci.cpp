#include "droid_stream_common.hpp"
#include "franka_realtime.hpp"
#include "franka_collision_diagnostics.hpp"
#include "droid_constraints.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <poll.h>
#include <unistd.h>
#include <pthread.h>

namespace {
using Clock = std::chrono::steady_clock;
constexpr double kFrankaFilterCutoffHz = 100.;
constexpr double kDroidMaxJointIncrement = .20;
double seconds() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
std::string jsonString(const std::string& value) {
  std::ostringstream out;out << '"';
  for(unsigned char c:value) {
    if(c=='"' || c=='\\') out << '\\' << c;
    else if(c<0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c);
    else out << c;
  }
  out << '"';return out.str();
}
void reportError(const std::string& message) {
  // Publish a complete fault record before the potentially large history dump.
  // Called outside the control callback, after the RT worker exits.
  std::cout << ("{\"ok\":false,\"error\":"+jsonString(message)+"}\n");
}

void activeGuard(const franka::RobotState& s) {
  if(stopped) throw std::runtime_error("Stop requested");
  droid::stateLimits(s);
}
void validateTarget(const Joints& expected,const Joints& target,const Joints& measured,
                    const Joints& lower,const Joints& upper) {
  for(size_t i=0;i<7;++i) {
    if(!std::isfinite(expected[i]) || !std::isfinite(target[i])) throw std::runtime_error("Nonfinite target");
    if(std::abs(target[i]-expected[i])>kDroidMaxJointIncrement+1e-6 ||
       std::abs(target[i]-measured[i])>kDroidMaxJointIncrement+.005001)
      throw std::runtime_error("Target exceeds DROID normalized action range");
    if(target[i]<lower[i] || target[i]>upper[i]) throw std::runtime_error("Joint limit margin");
  }
}
struct Command { Joints target{};double received=0;unsigned long id=0;bool hold=true; };
struct Feedback {franka::RobotState state{};double stamp=0,elapsed=0,max_error=0,max_velocity=0,min_success=1;unsigned long applied=0;unsigned safety_mask=0,saturation_mask=0,safety_seen=0,saturation_seen=0;double min_table_z=1e9;};
}

int main(int argc,char** argv) {
  std::cout<<std::setprecision(12)<<std::unitbuf;
  if(argc!=2 || std::string(argv[1])!="--session-confirmed") return 2;
  std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
  try {
    franka_rt::configureIo();
    franka::Robot robot("172.16.0.2",franka::RealtimeConfig::kEnforce);
    franka_rt::configureIo();
    franka_rt::lockMemory();
    auto s=robot.readOnce();idle(s);
    if(std::abs(s.m_ee)>1e-5) throw std::runtime_error("Desk tool mass nonzero; avoid double counting");
    for(size_t i=0;i<16;++i) if(std::abs(s.F_T_EE[i]-(i%5==0?1.:0.))>1e-6)
      throw std::runtime_error("Flange-to-EE transform differs from configured baseline");
    robot.setLoad(.9,{0,0,.057},{.002768,0,0,0,.003149,0,0,0,.000564});
    // The TCP command acknowledgement can precede its updated UDP state.
    // Wait for matching live readback before considering any motion command.
    const auto payload_deadline=Clock::now()+std::chrono::milliseconds(500);
    s=robot.readOnce();
    while(std::abs(s.m_total-.9)>1e-5 && Clock::now()<payload_deadline) {
      idle(s);
      s=robot.readOnce();
    }
    if(!std::isfinite(s.m_total) || std::abs(s.m_total-.9)>1e-5)
      throw std::runtime_error("Payload readback mismatch after 500 ms: "+std::to_string(s.m_total));
    idle(s);
    activeGuard(s);
    Joints lower{},upper{};
    tinyxml2::XMLDocument doc;
    const std::string urdf=robot.getRobotModel();
    auto model=robot.loadModel();
    franka::Model target_model(urdf);
    if(doc.Parse(urdf.c_str())!=tinyxml2::XML_SUCCESS) throw std::runtime_error("Invalid URDF");
    auto root=doc.FirstChildElement("robot");if(!root) throw std::runtime_error("Missing URDF root");
    for(size_t i=0;i<7;++i) {
      bool found=false;
      for(auto j=root->FirstChildElement("joint");j;j=j->NextSiblingElement("joint")) {
        const char* name=j->Attribute("name");
        if(name && std::string(name)=="joint"+std::to_string(i+1)) {
          auto lim=j->FirstChildElement("limit");
          if(!lim || lim->QueryDoubleAttribute("lower",&lower[i])!=tinyxml2::XML_SUCCESS ||
             lim->QueryDoubleAttribute("upper",&upper[i])!=tinyxml2::XML_SUCCESS) throw std::runtime_error("Joint limits missing");
          found=true;break;
        }
      }
      if(!found) throw std::runtime_error("Joint missing");
    }
    for(size_t j=0;j<7;++j) {
      if(droid::lower[j]<lower[j] || droid::upper[j]>upper[j])
        throw std::runtime_error("DROID joint limits exceed this robot's native model");
    }
    const auto modeled=model.pose(franka::Frame::kEndEffector,s);
    for(size_t axis=0;axis<3;++axis)
      if(std::abs(modeled[12+axis]-s.O_T_EE[12+axis])>.001)
        throw std::runtime_error("Native model FK disagrees with live flange position");
    lower=droid::lower;upper=droid::upper;
    std::cout << R"({"ready":true,"continuous_sessions_supported":true,"command_keepalive_supported":true,"motion_command_mode":"droid_joint_torques","application_smoothing":false,"application_motion_limits":null,"operating_motion_limits":null,"operating_profile":"droid_fr3","feedback_controller":"droid_hybrid_joint_cartesian_impedance","franka_rate_limiting":true,"franka_filter_cutoff_hz":100,"max_droid_joint_increment_rad":0.2,"constraint_profile":)"
              << droid::profile_json << ",\"realtime_config\":" << franka_rt::configJson();
    std::cout << ",\"joint_target_lower_rad\":";array(lower);
    std::cout << ",\"joint_target_upper_rad\":";array(upper);
    std::cout << ",\"collision_diagnostics\":"; collisionDiagnosticsJson(s);
    std::cout << ",\"state\":"; stateJson(s); std::cout << "}\n";
    std::string startline;std::getline(std::cin,startline);
    std::istringstream startin(startline);std::string op,extra;double duration=0;
    if(!(startin>>op>>duration) || op!="START" || startin>>extra || !std::isfinite(duration) || duration<0 || (duration>0 && duration<1))
      throw std::runtime_error("Explicit START duration 0 (continuous) or at least 1 second required");
    s=robot.readOnce();idle(s);activeGuard(s);
    std::array<double,7> native_torque{}; native_torque.fill(40.);
    std::array<double,6> native_force{}; native_force.fill(40.);
    robot.setCollisionBehavior(native_torque,native_torque,native_force,native_force);
    std::cout << "{\"native_collision_thresholds_applied\":true,\"all_axes\":40}\n";
    std::mutex command_mutex,state_mutex;
    Command command{s.q,seconds(),0,true};Feedback feedback{s,seconds()};feedback.min_table_z=s.O_T_EE[14];
    std::atomic<bool> stop_requested{false},done{false},worker_ready{false};std::string control_error,io_error;
    std::vector<franka::Record> control_error_records;
    std::thread controller([&] {
      try {
        franka_rt::configureControl();
        std::cout << "{\"realtime_thread\":" << franka_rt::threadJson() << "}\n";
        worker_ready=true;
        Command active{s.q,seconds(),0,true};bool first=true,stopping=false,holding=true;
        double elapsed=0,stop_since=0,max_error=0,max_velocity=0,min_success=1,min_table_z=1e9;
        unsigned long last_id=0;unsigned safety_seen=0,saturation_seen=0;
        Joints target=s.q;
        robot.control([&](const franka::RobotState& current,franka::Duration period)->franka::Torques {
          activeGuard(current);
          if(first) {first=false;target=current.q;for(double v:current.dq)if(std::abs(v)>.02)throw std::runtime_error("Initial state is not settled");}
          else {
            if(period.toSec()>.010) throw std::runtime_error("Control cycle gap exceeds 10 ms");
            elapsed+=period.toSec();
          }
          const auto jacobian=model.zeroJacobian(franka::Frame::kEndEffector,current);
          const double now=seconds();
          {std::unique_lock<std::mutex> lock(command_mutex,std::try_to_lock);if(lock.owns_lock()) active=command;}
          if(!stopping && (stop_requested || (duration>0 && elapsed>=duration) || now-active.received>.35)) {
            stopping=true;stop_since=elapsed;target=current.q;
            if(now-active.received>.35) control_error="Command watchdog expired";
          }
          if(active.id!=last_id && !stopping) {
            if(active.hold) {
              // Capture a fixed hold pose only when entering HOLD. Repeated
              // holds must not keep recapturing a compliance or payload drift.
              if(!holding) target=current.q;
              holding=true;
            } else {target=active.target;holding=false;}
            last_id=active.id;
          }
          auto result=droid::torques(current,target,jacobian,model.coriolis(current));
          safety_seen|=result.safety_mask;saturation_seen|=result.saturation_mask;
          min_table_z=std::min(min_table_z,current.O_T_EE[14]);
          bool settled=true;
          for(size_t j=0;j<7;++j) {
            max_error=std::max(max_error,std::abs(current.q[j]-target[j]));
            max_velocity=std::max(max_velocity,std::abs(current.dq[j]));
            if(std::abs(current.dq[j])>.02) settled=false;
          }
          if(elapsed>.1)min_success=std::min(min_success,current.control_command_success_rate);
          {std::unique_lock<std::mutex> lock(state_mutex,std::try_to_lock);
            if(lock.owns_lock()) feedback={current,now,elapsed,max_error,max_velocity,min_success,last_id,
                                         result.safety_mask,result.saturation_mask,safety_seen,saturation_seen,min_table_z};}
          if(stopping && elapsed-stop_since>2) throw std::runtime_error("Stopping timed out");
          franka::Torques output(result.torque);
          if(stopping && settled && elapsed-stop_since>.3)return franka::MotionFinished(output);
          return output;
        },true,kFrankaFilterCutoffHz);
      } catch(const franka::ControlException& e) {
        control_error=e.what();
        control_error_records=e.log;
      } catch(const std::exception& e) {control_error=e.what();}
      done=true;
    });
    while(!worker_ready && !done) std::this_thread::yield();
    // The pipe/JSON thread stays SCHED_OTHER on the housekeeping CPUs.
    std::string buffer;double next_output=0;unsigned long last_received_id=0;
    try {
      while(!done) {
        Feedback snapshot;{std::lock_guard<std::mutex> lock(state_mutex);snapshot=feedback;}
        if(seconds()>=next_output) {
          std::cout<<"{\"state\":";stateJson(snapshot.state);
          std::cout<<",\"collision_diagnostics\":";collisionDiagnosticsJson(snapshot.state);
          std::cout<<",\"stamp\":"<<snapshot.stamp<<",\"elapsed\":"<<snapshot.elapsed
                   <<",\"applied\":"<<snapshot.applied<<",\"max_joint_error\":"<<snapshot.max_error
                   <<",\"max_joint_velocity\":"<<snapshot.max_velocity<<",\"min_success\":"<<snapshot.min_success
                   <<",\"safety_feedback_mask\":"<<snapshot.safety_mask<<",\"torque_saturation_mask\":"<<snapshot.saturation_mask
                   <<",\"safety_feedback_seen_mask\":"<<snapshot.safety_seen<<",\"torque_saturation_seen_mask\":"<<snapshot.saturation_seen
                   <<",\"minimum_flange_z_m\":"<<snapshot.min_table_z<<"}\n";
          next_output=seconds()+.02;
        }
        pollfd fd{STDIN_FILENO,POLLIN,0};int ready=poll(&fd,1,2);
        if(ready<0) {if(stopped) break;throw std::runtime_error("stdin poll failed");}
        if(ready>0 && (fd.revents&(POLLIN|POLLHUP))) {
          char bytes[4096];ssize_t n=read(STDIN_FILENO,bytes,sizeof(bytes));
          if(n<=0) {stop_requested=true;continue;}
          buffer.append(bytes,n);
          if(buffer.size()>8192) throw std::runtime_error("Command buffer overflow");
          size_t newline;
          while((newline=buffer.find('\n'))!=std::string::npos) {
            std::string line=buffer.substr(0,newline);buffer.erase(0,newline+1);
            if(line=="STOP") {stop_requested=true;continue;}
            Command update;std::istringstream in(line);std::string kind;
            if(!(in>>kind>>update.id) || update.id<=last_received_id) throw std::runtime_error("Invalid command sequence");
            if(kind=="KEEPALIVE") {
              if(in>>extra) throw std::runtime_error("Extra keepalive data");
              last_received_id=update.id;
              // Refresh liveness only. The validated target, HOLD mode and
              // acknowledged action ID remain unchanged; watchdog stays 350 ms.
              {std::lock_guard<std::mutex> lock(command_mutex);command.received=seconds();}
              continue;
            }
            update.hold=(kind=="HOLD");
            if(kind=="TARGET") {
              Joints expected{};
              for(double& v:expected) if(!(in>>v)) throw std::runtime_error("Expected joints missing");
              for(double& v:update.target) if(!(in>>v)) throw std::runtime_error("Target joints missing");
              validateTarget(expected,update.target,snapshot.state.q,lower,upper);
              if(!droid::targetPath(snapshot.state,update.target,target_model)) {
                update.hold=true;
                std::cout<<"{\"target_rejected\":true,\"reason\":\"droid_workspace\",\"id\":"<<update.id<<"}\n";
              }
            } else if(kind!="HOLD") throw std::runtime_error("Unknown command");
            if(in>>extra) throw std::runtime_error("Extra command data");
            update.received=seconds();last_received_id=update.id;
            {std::lock_guard<std::mutex> lock(command_mutex);command=update;}
          }
        }
      }
    } catch(const std::exception& e) {io_error=e.what();stop_requested=true;}
    stop_requested=true;controller.join();
    // Fault notification must precede the history dump, or the client can hit
    // its feedback deadline while the underlying reflex is still being logged.
    if(!control_error.empty()) reportError(control_error);
    else if(!io_error.empty()) reportError(io_error);
    // Emit libfranka's pre-stop records only after the real-time worker exits.
    // Keep these separate from live feedback so old timestamps cannot be used
    // for policy observations.
    for(const auto& record:control_error_records) {
      std::cout << "{\"control_error_record\":";
      collisionDiagnosticsJson(record.state);
      std::cout << "}\n";
    }
    if(!control_error.empty()) throw std::runtime_error(control_error);
    if(!io_error.empty()) throw std::runtime_error(io_error);
    auto after=robot.readOnce();activeGuard(after);
    std::cout<<"{\"complete\":true,\"state\":";stateJson(after);std::cout<<"}\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"STREAM_FCI_STOPPED: "<<e.what()<<'\n';
    reportError(e.what());return 1;
  }
}
