// Opt-in localhost training bridge, compiled only as part of the Gazebo SITL module.
// All blocking IPC lives in its own pthread, never the rate-control work queue.
#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/actuators.pb.h>
#include <uORB/Subscription.hpp>
#include <uORB/Publication.hpp>
#include <uORB/topics/vehicle_rates_setpoint.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/offboard_control_mode.h>
#include <uORB/topics/goto_setpoint.h>
#include <uORB/topics/takeoff_status.h>
#include <uORB/topics/vehicle_thrust_setpoint.h>
#include <uORB/topics/vehicle_torque_setpoint.h>
#include <uORB/topics/actuator_motors.h>
#include <drivers/drv_hrt.h>
#include <mutex>
#include <array>
#include <map>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <px4_platform_common/log.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <pthread.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstdio>

extern "C" int mc_rate_control_main(int argc, char *argv[]);
extern "C" int control_allocator_main(int argc, char *argv[]);

namespace {
using Wall = std::chrono::steady_clock;
struct Bridge {
    std::string world;
    int port;
    gz::transport::Node node;
    std::atomic<uint64_t> clock_us{0};
    uORB::Subscription rates_sub{ORB_ID(vehicle_angular_velocity)};
    bool native = std::getenv("PX4_RL_NATIVE_OUTER") != nullptr;
    bool baseline = std::getenv("PX4_RL_NATIVE_BASELINE") != nullptr;
    bool native_torque = std::getenv("PX4_RL_NATIVE_TORQUE") != nullptr;
    std::atomic<bool> policy_control{false};
    uORB::Subscription setpoint_sub{ORB_ID(vehicle_rates_setpoint)};
    uORB::Subscription thrust_forward_sub{ORB_ID(vehicle_rates_setpoint)};
    uORB::Subscription local_sub{ORB_ID(vehicle_local_position)};
    uORB::Subscription attitude_sp_sub{ORB_ID(vehicle_attitude_setpoint)};
    uORB::Subscription mode_sub{ORB_ID(vehicle_control_mode)};
    uORB::Subscription status_sub{ORB_ID(vehicle_status)};
    uORB::Subscription takeoff_sub{ORB_ID(takeoff_status)};
    uORB::Publication<offboard_control_mode_s> offboard_pub{ORB_ID(offboard_control_mode)};
    uORB::Publication<goto_setpoint_s> goal_pub{ORB_ID(goto_setpoint)};
    uORB::Publication<vehicle_command_s> command_pub{ORB_ID(vehicle_command)};
    uORB::Publication<vehicle_thrust_setpoint_s> thrust_pub{ORB_ID(vehicle_thrust_setpoint)};
    uORB::Publication<vehicle_torque_setpoint_s> torque_pub{ORB_ID(vehicle_torque_setpoint)};
    uORB::Subscription allocated_sub{ORB_ID(actuator_motors)};
    std::mutex torque_mutex;
    float policy_torque[3]{};
    bool torque_valid{false};
    uint64_t action_sample_us{0};
    std::atomic<uint64_t> torque_interval_end_us{0};
    struct ForwardedThrust { std::array<float, 3> xyz; uint64_t reference_us; };
    std::map<uint64_t, ForwardedThrust> forwarded_thrust;
    std::mutex goal_mutex;
    goto_setpoint_s goal{};
    bool goal_valid{false};
    uint64_t heartbeat_us{0};
    std::mutex motor_mutex;
    double native_motor_speed[4]{};
    void native_motors(const gz::msgs::Actuators &msg) {
        std::lock_guard<std::mutex> lock(motor_mutex);
        if (msg.velocity_size() >= 4)
            for (int i = 0; i < 4; ++i) native_motor_speed[i] = msg.velocity(i);
    }

    void publish_goal() {
        std::lock_guard<std::mutex> lock(goal_mutex);
        if (!native || !goal_valid) return;
        const auto now = hrt_absolute_time();
        offboard_control_mode_s mode{};
        mode.timestamp = now; mode.position = true;
        offboard_pub.publish(mode);
        goal.timestamp = now;
        goal_pub.publish(goal);
        if (policy_control.load() && !native_torque) {
            // Normally mc_rate_control forwards this to land detection. Keep
            // the same desired-thrust status while PPO owns actuator outputs.
            vehicle_rates_setpoint_s sp{}; thrust_forward_sub.copy(&sp);
            vehicle_thrust_setpoint_s thrust{};
            thrust.timestamp = now; thrust.timestamp_sample = now;
            for (int i = 0; i < 3; ++i) thrust.xyz[i] = sp.thrust_body[i];
            thrust_pub.publish(thrust);
        }
    }
    void publish_native_torque(uint64_t sample) {
        std::lock_guard<std::mutex> lock(torque_mutex);
        if (!native_torque || !policy_control.load() || !torque_valid) return;
        vehicle_rates_setpoint_s sp{};
        thrust_forward_sub.copy(&sp);
        const auto now = hrt_absolute_time();
        vehicle_thrust_setpoint_s thrust{};
        thrust.timestamp = now; thrust.timestamp_sample = sample;
        for (int i = 0; i < 3; ++i) thrust.xyz[i] = sp.thrust_body[i];
        vehicle_torque_setpoint_s torque{};
        torque.timestamp = now; torque.timestamp_sample = sample;
        for (int i = 0; i < 3; ++i) torque.xyz[i] = policy_torque[i];
        forwarded_thrust[sample] = {{sp.thrust_body[0], sp.thrust_body[1], sp.thrust_body[2]}, sp.timestamp};
        while (forwarded_thrust.size() > 200) forwarded_thrust.erase(forwarded_thrust.begin());
        // The allocator is driven by the torque callback: matching thrust first.
        thrust_pub.publish(thrust);
        torque_pub.publish(torque);
    }
    void vehicle_command(uint32_t id, float p1, float p2) {
        vehicle_command_s cmd{};
        cmd.timestamp = hrt_absolute_time(); cmd.command = id;
        cmd.param1 = p1; cmd.param2 = p2;
        cmd.target_system = 0; cmd.target_component = 1; // SITL instance IDs change MAV_SYS_ID.
        cmd.source_system = 1; cmd.source_component = 191;
        cmd.from_external = false; // Internal, simulation-only air-start bootstrap.
        command_pub.publish(cmd);
    }
    bool native_init() {
        if (!native) return true;
        const auto deadline = Wall::now() + std::chrono::seconds(120);
        uint64_t started = 0, last_command = 0, stable_since = 0;
        while (Wall::now() < deadline) {
            vehicle_local_position_s local{}; local_sub.copy(&local);
            gz::msgs::Boolean request;
            gz::msgs::Double_V truth;
            bool result = false;
            const bool truth_ok = node.Request("/rate_training/snapshot", request, 500, truth, result)
                && result && truth.data_size() == 25 && std::abs(truth.data(1) - 1.) < .01;
            if (truth_ok && local.xy_valid && local.z_valid && local.v_xy_valid && local.v_z_valid) {
                if (!started) {
                    std::lock_guard<std::mutex> lock(goal_mutex);
                    goal.position[0] = local.x; goal.position[1] = local.y;
                    // EKF local origin differs from Gazebo ground/world zero.
                    goal.position[2] = local.z - float(truth.data(14)) - 5.f;
                    goal.flag_control_heading = true; goal.heading = local.heading;
                    goal_valid = true; started = hrt_absolute_time();
                }
                auto now = hrt_absolute_time();
                if (now > started + 1500000 && now > last_command + 1000000) {
                    vehicle_status_s status{}; status_sub.copy(&status);
                    if (status.nav_state != vehicle_status_s::NAVIGATION_STATE_OFFBOARD)
                        vehicle_command(vehicle_command_s::VEHICLE_CMD_DO_SET_MODE, 1.f, 6.f);
                    else if (status.arming_state != vehicle_status_s::ARMING_STATE_ARMED)
                        vehicle_command(vehicle_command_s::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.f, 0.f);
                    last_command = now;
                }
                vehicle_control_mode_s mode{}; mode_sub.copy(&mode);
                vehicle_rates_setpoint_s sp{}; setpoint_sub.copy(&sp);
                takeoff_status_s takeoff{}; takeoff_sub.copy(&takeoff);
                if (mode.flag_armed && mode.flag_control_offboard_enabled && mode.flag_control_position_enabled
                    && mode.flag_control_attitude_enabled && sp.timestamp > started
                    && now >= sp.timestamp && now - sp.timestamp <= 50000
                    && std::isfinite(sp.roll) && std::isfinite(sp.pitch) && std::isfinite(sp.yaw)
                    && std::isfinite(sp.thrust_body[2]) && sp.thrust_body[2] < 0.f
                    && takeoff.takeoff_state == takeoff_status_s::TAKEOFF_STATE_FLIGHT
                    && std::abs(truth.data(14) + 5.) < .2
                    && std::hypot(truth.data(19), truth.data(20)) < .2
                    && std::abs(truth.data(21)) < .2
                    && std::abs(truth.data(22)) < .0872664626
                    && std::abs(truth.data(23)) < .0872664626
                    && std::abs(truth.data(24)) < .0872664626) {
                    if (!stable_since) stable_since = now;
                    if (now >= stable_since + 2000000) {
                        PX4_INFO("RL native ready: reached 5m, stable 2s; height=%.3f thrust=%.3f",
                            -truth.data(14), double(sp.thrust_body[2]));
                        return true;
                    }
                } else stable_since = 0;
            } else stable_since = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        vehicle_local_position_s local{}; local_sub.copy(&local);
        vehicle_status_s status{}; status_sub.copy(&status);
        vehicle_rates_setpoint_s sp{}; setpoint_sub.copy(&sp);
        PX4_ERR("RL native init failed: local=%d/%d nav=%u armed=%u thrust=%.3f", local.xy_valid, local.z_valid,
                unsigned(status.nav_state), unsigned(status.arming_state), double(sp.thrust_body[2]));
        return false;
    }

    void clock(const gz::msgs::Clock &c) {
        clock_us.store(uint64_t(c.sim().sec()) * 1000000 + c.sim().nsec() / 1000);
        if (native && clock_us.load() >= heartbeat_us + 10000) {
            heartbeat_us = clock_us.load(); publish_goal();
        }
        // Do not publish the previous action at an interval's final boundary:
        // that timestamp belongs exclusively to the next policy action.
        if (clock_us.load() < torque_interval_end_us.load()) publish_native_torque(clock_us.load());
    }
    bool control(bool paused, uint32_t steps = 0) {
        gz::msgs::WorldControl request;
        request.set_pause(paused);
        request.set_multi_step(steps);
        gz::msgs::Boolean response;
        bool result = false;
        return node.Request("/world/" + world + "/control", request, 5000, response, result)
            && result && response.data();
    }
    bool latch(uint64_t seq, const float *motors) {
        gz::msgs::Double_V request;
        request.add_data(seq);
        for (unsigned i = 0; i < 4; ++i) request.add_data(motors[i]);
        gz::msgs::Boolean response;
        bool result = false;
        return node.Request("/rate_training/action", request, 5000, response, result)
            && result && response.data();
    }
    bool latch_torque(uint64_t seq, const float *torque) {
        const uint64_t sample = clock_us.load();
        {
            std::lock_guard<std::mutex> lock(torque_mutex);
            for (int i = 0; i < 3; ++i) policy_torque[i] = torque[i];
            torque_valid = true;
            action_sample_us = sample;
        }
        publish_native_torque(sample);
        gz::msgs::Double_V request;
        request.add_data(seq); request.add_data(sample);
        gz::msgs::Boolean response;
        bool result = false;
        bool ok = false;
        const auto deadline = Wall::now() + std::chrono::seconds(3);
        do {
            ok = node.Request("/rate_training/native_action", request, 500, response, result)
                && result && response.data();
            if (ok) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (Wall::now() < deadline);
        if (!ok) {
            actuator_motors_s allocated{}; allocated_sub.copy(&allocated);
            PX4_ERR("RL torque ACK failed: action=%llu alloc_sample=%llu now=%llu controls=%.3f/%.3f/%.3f/%.3f",
                (unsigned long long)sample, (unsigned long long)allocated.timestamp_sample,
                (unsigned long long)hrt_absolute_time(), double(allocated.control[0]), double(allocated.control[1]),
                double(allocated.control[2]), double(allocated.control[3]));
        }
        return ok;
    }
    std::string state(uint64_t seq, uint64_t expected = 0) {
        vehicle_angular_velocity_s rates{};
        gz::msgs::Boolean request;
        gz::msgs::Double_V truth, noise, allocation;
        const auto deadline = Wall::now() + std::chrono::seconds(5);
        bool fresh = false;
        do {
            rates_sub.copy(&rates);
            bool truth_result = false, noise_result = false;
            const bool truth_ok = node.Request("/rate_training/snapshot", request, 500, truth, truth_result)
                && truth_result && truth.data_size() == 25
                && std::abs(truth.data(0) - 2) < 0.01 && std::abs(truth.data(1) - 1) < 0.01;
            const bool noise_ok = node.Request("/rate_training/noise", request, 500, noise, noise_result)
                && noise_result && noise.data_size() == 10;
            fresh = truth_ok && noise_ok && rates.timestamp_sample > 0;
            if (fresh && native_torque) {
                bool allocation_result = false;
                fresh = node.Request("/rate_training/native_snapshot", request, 500, allocation, allocation_result)
                    && allocation_result && allocation.data_size() == 8
                    && std::abs(allocation.data(0) - truth.data(2)) < .5
                    && std::abs(allocation.data(1) - truth.data(3)) < .5;
                if (fresh && seq) {
                    fresh = std::abs(allocation.data(3) - double(action_sample_us)) < .5
                        && allocation.data(2) >= double(action_sample_us);
                }
            }
            // Bootstrap holds the body while the actual rotor filters settle.
            // The first policy observation must agree with the preloaded PWM.
            if (fresh && !expected && !native) {
                for (int i = 0; i < 4; ++i)
                    fresh = fresh && std::abs(truth.data(8 + i) - truth.data(4 + i)) <= 0.005;
            }
            if (fresh && expected) {
                fresh = rates.timestamp_sample == expected && std::abs(truth.data(2) - double(expected)) < 0.5
                    && std::abs(noise.data(0) - double(expected)) < 0.5
                    && std::abs(truth.data(3) - double(seq)) < 0.5;
            }
            if (fresh) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (Wall::now() < deadline);
        if (!fresh) {
            std::ostringstream error;
            error << "{\"error\":\"synchronized feedback timeout: expected=" << expected
                << " gyro=" << rates.timestamp_sample
                << " truth=" << (truth.data_size() > 2 ? truth.data(2) : -1)
                << " valid=" << (truth.data_size() > 1 ? truth.data(1) : -1)
                << " noise=" << (noise.data_size() ? noise.data(0) : -1) << "\"}\n";
            return error.str();
        }
        std::ostringstream out;
        out << std::setprecision(10) << "{\"seq\":" << seq << ",\"sim_us\":" << clock_us.load()
            << ",\"sample_us\":" << rates.timestamp_sample
            << ",\"source_sample_us\":" << rates.timestamp_sample
            << ",\"truth_us\":" << uint64_t(truth.data(2))
            << ",\"action_seq\":" << uint64_t(truth.data(3))
            << ",\"snapshot_valid\":true,\"snapshot_version\":2,\"rates\":["
            << rates.xyz[0] << ',' << rates.xyz[1] << ',' << rates.xyz[2] << ']';
        const auto array = [&](const char *name, int start, int count) {
            out << ",\"" << name << "\":[";
            for (int i = 0; i < count; ++i) {
                if (i) out << ',';
                out << truth.data(start + i);
            }
            out << ']';
        };
        array("applied_pwm", 4, 4);
        if (native_torque) {
            array("applied_speed_fraction", 4, 4);
            out << ",\"allocated_pwm\":[";
            for (int i = 0; i < 4; ++i) { if (i) out << ','; out << allocation.data(4 + i); }
            out << "]";
            std::lock_guard<std::mutex> lock(torque_mutex);
            out << ",\"applied_torque\":[" << policy_torque[0] << ',' << policy_torque[1] << ',' << policy_torque[2] << ']'
                << ",\"allocator_action_sample_us\":" << action_sample_us
                << ",\"allocator_first_sample_us\":" << uint64_t(allocation.data(3))
                << ",\"allocator_last_sample_us\":" << uint64_t(allocation.data(2))
                << ",\"allocator_action_matched\":" << (seq && std::abs(allocation.data(3) - double(action_sample_us)) < .5 ? "true" : "false");
            const auto thrust = forwarded_thrust.find(uint64_t(allocation.data(2)));
            if (thrust != forwarded_thrust.end()) {
                out << ",\"forwarded_thrust_body\":[" << thrust->second.xyz[0] << ',' << thrust->second.xyz[1] << ',' << thrust->second.xyz[2] << ']'
                    << ",\"forwarded_thrust_sample_us\":" << thrust->first
                    << ",\"forwarded_thrust_reference_us\":" << thrust->second.reference_us;
            }
        }
        array("rotor_speed_fraction", 8, 4);
        // base_link / pivot coordinates, not the included model frame offset.
        array("position_ned", 12, 3);
        array("q_ned_frd", 15, 4);
        array("linear_velocity_ned", 19, 3);
        array("rates_true", 22, 3);
        out << ",\"gyro_noise_flu\":[";
        for (int i = 0; i < noise.data_size(); ++i) {
            if (i) out << ',';
            out << noise.data(i);
        }
        out << ']';
        if (native) {
            vehicle_rates_setpoint_s sp{}; vehicle_local_position_s local{};
            vehicle_attitude_setpoint_s asp{}; vehicle_control_mode_s mode{};
            setpoint_sub.copy(&sp); local_sub.copy(&local); attitude_sp_sub.copy(&asp); mode_sub.copy(&mode);
            out << ",\"px4_rate_target\":[" << sp.roll << ',' << sp.pitch << ',' << sp.yaw << ']'
                << ",\"px4_thrust_body_z\":" << sp.thrust_body[2]
                << ",\"px4_reference_us\":" << sp.timestamp
                << ",\"px4_now_us\":" << hrt_absolute_time()
                << ",\"px4_policy_owns_motors\":" << (policy_control.load() && !native_torque ? "true" : "false")
                << ",\"px4_policy_owns_torque\":" << (native_torque && policy_control.load() ? "true" : "false")
                << ",\"px4_native_allocator\":" << (native_torque && policy_control.load() ? "true" : "false")
                << ",\"px4_local_position\":[" << local.x << ',' << local.y << ',' << local.z << ']'
                << ",\"px4_local_sample_us\":" << local.timestamp_sample
                << ",\"px4_heading\":" << local.heading
                << ",\"px4_attitude_target\":[" << asp.q_d[0] << ',' << asp.q_d[1] << ',' << asp.q_d[2] << ',' << asp.q_d[3] << ']'
                << ",\"px4_position_control\":" << (mode.flag_armed && mode.flag_control_position_enabled && mode.flag_control_attitude_enabled && mode.flag_control_offboard_enabled ? "true" : "false");
            { std::lock_guard<std::mutex> lock(motor_mutex);
              out << ",\"px4_baseline_motor_speed\":[";
              for (int i = 0; i < 4; ++i) { if (i) out << ','; out << native_motor_speed[i]; }
              out << ']'; }
        }
        out << "}\n";
        return out.str();
    }
    void run() {
        node.Subscribe("/world/" + world + "/clock", &Bridge::clock, this);
        if (native) node.Subscribe("/x500_rl_0/command/motor_speed", &Bridge::native_motors, this);
        const int server = socket(AF_INET, SOCK_STREAM, 0);
        int yes = 1;
        setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) || listen(server, 1)) {
            PX4_ERR("RL bridge bind failed"); close(server); return;
        }
        PX4_INFO("RL bridge ready on localhost:%d", port);
        const int client = accept(server, nullptr, nullptr);
        close(server);
        if (client < 0) return;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        FILE *stream = fdopen(client, "r+");
        if (!stream) { close(client); return; }
        setvbuf(stream, nullptr, _IOLBF, 0);
        char line[512];
        uint64_t sequence = 0;
        bool initialized = false;
        while (fgets(line, sizeof(line), stream)) {
            std::istringstream input(line);
            std::string command;
            input >> command;
            std::string reply;
            if (command == "INIT" && !initialized) {
                if (!native_init()) { fputs("{\"error\":\"PX4 native outer initialization failed\"}\n", stream); fflush(stream); break; }
                if (native && !baseline) {
                    policy_control.store(true);
                    char rate_name[] = "mc_rate_control", allocator_name[] = "control_allocator", stop[] = "stop";
                    char *rate_args[] = {rate_name, stop};
                    char *allocator_args[] = {allocator_name, stop};
                    if ((!native_torque && control_allocator_main(2, allocator_args) != 0) || mc_rate_control_main(2, rate_args) != 0) {
                        fputs("{\"error\":\"failed to stop native inner loop\"}\n", stream); fflush(stream); break;
                    }
                    PX4_INFO("RL owns %s; native rate controller stopped; allocator %s, outer loops retained",
                        native_torque ? "torque" : "motors", native_torque ? "retained" : "stopped");
                }
                if (native_torque) {
                    // Stop the callback-driven rate module while sensors still
                    // advance, then pause before reading same-time snapshots.
                    if (!control(true)) { fputs("{\"error\":\"native torque bootstrap pause failed\"}\n", stream); fflush(stream); break; }
                    auto last = clock_us.load();
                    for (int i = 0; i < 100; ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        auto now = clock_us.load();
                        if (now == last) break;
                        last = now;
                    }
                }
                if (native) {
                    gz::msgs::Boolean request, response;
                    bool result = false;
                    if (!node.Request("/rate_training/takeover", request, 5000, response, result)
                        || !result || !response.data()) {
                        fputs("{\"error\":\"native motor takeover failed\"}\n", stream); fflush(stream); break;
                    }
                }
                // Wait for the normal PX4 sensor chain before pausing physics.
                reply = state(0);
                if (reply.find("error") == std::string::npos && control(true)) {
                    auto last = clock_us.load();
                    for (int i = 0; i < 100; ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        auto now = clock_us.load();
                        if (now == last) break;
                        last = now;
                    }
                    if (native_torque) {
                        // Separate the first RL sample from all pre-stop PID samples.
                        // This 1ms handover interval is outside episode time/steps.
                        const uint64_t end = clock_us.load() + 1000;
                        if (!control(true, 1)) { fputs("{\"error\":\"torque handover step failed\"}\n", stream); fflush(stream); break; }
                        const auto deadline = Wall::now() + std::chrono::seconds(5);
                        while (clock_us.load() < end && Wall::now() < deadline)
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        if (clock_us.load() != end) { fputs("{\"error\":\"torque handover time mismatch\"}\n", stream); fflush(stream); break; }
                    }
                    reply = state(0, clock_us.load());
                    initialized = reply.find("error") == std::string::npos;
                } else if (reply.find("error") == std::string::npos) reply = "{\"error\":\"initialization pause failed\"}\n";
            } else if ((command == "STEP" || command == "STEP_TORQUE") && initialized) {
                uint64_t seq = 0;
                unsigned steps = 0;
                float motors[4]{};
                input >> seq >> steps >> motors[0] >> motors[1] >> motors[2];
                if (!native_torque) input >> motors[3];
                bool valid = bool(input) && seq == sequence + 1 && steps >= 1 && steps <= 100;
                valid = valid && (native_torque ? command == "STEP_TORQUE" : command == "STEP");
                for (int i = 0; i < (native_torque ? 3 : 4); ++i)
                    valid = valid && std::isfinite(motors[i]) && motors[i] >= (native_torque ? -1 : 0) && motors[i] <= 1;
                if (valid && native_torque) torque_interval_end_us.store(clock_us.load() + uint64_t(steps) * 1000);
                if (!valid) reply = "{\"error\":\"invalid action or sequence\"}\n";
                else if (!(native_torque ? latch_torque(seq, motors) : latch(seq, motors))) reply = "{\"error\":\"action latch timeout\"}\n";
                else {
                    const uint64_t end = clock_us.load() + uint64_t(steps) * 1000;
                    if (!control(true, steps)) reply = "{\"error\":\"world step failed\"}\n";
                    else {
                        const auto deadline = Wall::now() + std::chrono::seconds(5);
                        while (clock_us.load() < end && Wall::now() < deadline)
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        if (clock_us.load() != end) reply = "{\"error\":\"simulation step mismatch\"}\n";
                        else { sequence = seq; reply = state(sequence, end); }
                    }
                }
            } else if (command == "GOAL" && initialized && native) {
                float x, y, z, yaw;
                input >> x >> y >> z >> yaw;
                if (!input || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(yaw))
                    reply = "{\"error\":\"invalid position goal\"}\n";
                else {
                    { std::lock_guard<std::mutex> lock(goal_mutex);
                      goal.position[0] = x; goal.position[1] = y; goal.position[2] = z;
                      goal.heading = yaw; goal.flag_control_heading = true; }
                    publish_goal(); reply = state(sequence, clock_us.load());
                }
            } else if (command == "STATE" && initialized) reply = state(sequence, clock_us.load());
            else reply = "{\"error\":\"unknown command or not initialized\"}\n";
            if (fputs(reply.c_str(), stream) < 0 || fflush(stream) != 0) break;
        }
        control(true);
        fclose(stream);
    }
};
void *entry(void *arg) { static_cast<Bridge *>(arg)->run(); return nullptr; }
}

void start_rl_bridge(const std::string &world) {
    const char *port = std::getenv("PX4_RL_PORT");
    if (!port) return;
    auto *bridge = new Bridge;
    bridge->world = world;
    bridge->port = std::atoi(port);
    pthread_t thread;
    if (pthread_create(&thread, nullptr, entry, bridge) == 0) pthread_detach(thread);
    else PX4_ERR("RL bridge thread creation failed");
}
