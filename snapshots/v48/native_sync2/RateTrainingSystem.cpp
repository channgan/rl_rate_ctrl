// Simulation-only action latch. The motor model reads the Actuators component
// before transport commands, so the stock PX4 output cannot overwrite RL actions.
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/components/Actuators.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/actuators.pb.h>
#include <cstdlib>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <stdexcept>

class RateTrainingSystem final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
    public gz::sim::ISystemConfigurePriority, public gz::sim::ISystemPostUpdate {
    gz::sim::Model model{gz::sim::kNullEntity};
    gz::sim::Link base{gz::sim::kNullEntity};
    gz::transport::Node node;
    gz::transport::Node::Publisher sync_event;
    void notify_sync() {if(closed_loop_sync && sync_event.Valid()){gz::msgs::Boolean msg;msg.set_data(true);sync_event.Publish(msg);}}
    std::mutex mutex;
    std::condition_variable applied;
    std::array<double, 4> motors{0, 0, 0, 0};
    bool closed_loop_sync{std::getenv("RATE_RL_CLOSED_LOOP_SYNC") != nullptr};
    uint64_t sync_received{0}, sync_applied{0};
    bool sync_received_valid{false}, sync_applied_valid{false};
    uint64_t requested{0}, committed{0};
    bool active{false};
    bool native_bootstrap{std::getenv("PX4_RL_NATIVE_OUTER") != nullptr};
    bool native_baseline{std::getenv("PX4_RL_NATIVE_BASELINE") != nullptr};
    bool native_torque{std::getenv("PX4_RL_NATIVE_TORQUE") != nullptr};
    uint64_t native_sample{0}, expected_native_sample{0}, committed_native_sample{0}, first_native_sample{0};
    std::array<double, 4> allocated{}, applied_motors{}, applied_allocated{};
    std::array<double, 8> native_snapshot{};
    double max_speed{1000}, hover{0.77}, slowdown{10};
    FILE *physics_capture{nullptr};
    double captured_time{-1};
    double spawn_height{3};
    std::array<gz::sim::Entity, 4> rotors{};
    // Versioned complete PostUpdate snapshot: version, valid, sim_us, action_seq,
    // applied PWM[4], rotor fractions[4], NED position[3], FRD->NED q[wxyz],
    // NED linear velocity[3], FRD angular velocity[3]. Never mix update times.
    std::array<double, 25> snapshot{};
public:
    gz::sim::System::PriorityType ConfigurePriority() override { return -100; }
    void Configure(const gz::sim::Entity &entity, const std::shared_ptr<const sdf::Element> &sdf,
        gz::sim::EntityComponentManager &ecm, gz::sim::EventManager &) override {
        if (const char *path = std::getenv("RATE_RL_PHYSICS_CAPTURE")) {
            physics_capture = std::fopen(path, "wx");
            if (!physics_capture) { throw std::runtime_error("Cannot create unique physics capture"); }
            std::fwrite("PHYS25V2", 1, 8, physics_capture);
        }
        model = gz::sim::Model(entity);
        base = gz::sim::Link(model.LinkByName(ecm, "base_link"));
        base.EnableVelocityChecks(ecm);
        if (sdf->HasElement("max_speed")) max_speed = sdf->Get<double>("max_speed");
        if (sdf->HasElement("hover")) hover = sdf->Get<double>("hover");
        if (sdf->HasElement("slowdown")) slowdown = sdf->Get<double>("slowdown");
        if (sdf->HasElement("spawn_height")) spawn_height = sdf->Get<double>("spawn_height");
        motors.fill(hover);
        if (native_bootstrap) motors.fill(0);
        for (int i = 0; i < 4; ++i) rotors[i] = model.JointByName(ecm, "rotor_" + std::to_string(i) + "_joint");
        node.Advertise("/rate_training/action", &RateTrainingSystem::Action, this);
        node.Advertise("/rate_training/snapshot", &RateTrainingSystem::Snapshot, this);
        node.Advertise("/rate_training/closed_loop_ack", &RateTrainingSystem::ClosedLoopAck, this);
        if(closed_loop_sync) sync_event=node.Advertise<gz::msgs::Boolean>("/rate_training/closed_loop_event");
        node.Advertise("/rate_training/takeover", &RateTrainingSystem::Takeover, this);
        node.Advertise("/rate_training/native_action", &RateTrainingSystem::NativeAction, this);
        node.Advertise("/rate_training/native_snapshot", &RateTrainingSystem::NativeSnapshot, this);
        if (native_bootstrap)
            node.Subscribe("/x500_rl_0/command/motor_speed", &RateTrainingSystem::NativeMotors, this);
    }
    void NativeMotors(const gz::msgs::Actuators &message) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!native_bootstrap || (active && !native_torque && !native_baseline) || message.velocity_size() < 4) return;
        for (int i = 0; i < 4; ++i)
            if (!std::isfinite(message.velocity(i)) || message.velocity(i) < 0 || message.velocity(i) > max_speed) return;
        if (native_torque) {
            const auto &header = message.header();
            const uint64_t sample = uint64_t(header.stamp().sec()) * 1000000 + header.stamp().nsec() / 1000;
            if (!sample || sample < native_sample) return;
            bool valid = false;
            for (const auto &data : header.data()) {
                if (data.key() != "rl_allocated_controls" || data.value_size() != 4) continue;
                std::array<double, 4> values{};
                for (int i = 0; i < 4; ++i) {
                    char *end = nullptr;
                    values[i] = std::strtod(data.value(i).c_str(), &end);
                    if (!end || *end || !std::isfinite(values[i]) || values[i] < 0 || values[i] > 1) return;
                }
                allocated = values;
                valid = true;
            }
            if (!valid) return;
            native_sample = sample;
        }
        bool notify_received=false;
        if (closed_loop_sync) {
            const uint64_t before=sync_received;const bool valid_before=sync_received_valid;
            const auto &header=message.header();
            const uint64_t sample=uint64_t(header.stamp().sec())*1000000+header.stamp().nsec()/1000;
            // Metadata describes this exact message. Never tag older motor values
            // with a newer timestamp and never change the accepted motor values.
            sync_received_valid=sample>0 && sample>=sync_received;
            sync_received=sample;notify_received=(sample!=before || valid_before!=sync_received_valid);
        }
        for (int i = 0; i < 4; ++i) motors[i] = message.velocity(i) / max_speed;
        if(notify_received) notify_sync();
    }
    bool Takeover(const gz::msgs::Boolean &, gz::msgs::Boolean &response) {
        std::lock_guard<std::mutex> lock(mutex);
        // Freeze the last actual native outputs until policy action 1 arrives.
        // Do not reset rotor dynamics or the action sequence.
        active = true;
        response.set_data(true);
        return true;
    }
    bool ClosedLoopAck(const gz::msgs::Boolean &, gz::msgs::Double_V &response) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!closed_loop_sync || !native_baseline || native_torque) return false;
        // No blocking wait here: motor delivery uses this transport too.
        response.add_data(snapshot[2]); response.add_data(snapshot[3]);
        response.add_data(double(sync_received)); response.add_data(double(sync_applied));
        response.add_data(sync_received_valid ? 1. : 0.); response.add_data(sync_applied_valid ? 1. : 0.);
        response.add_data(snapshot[1]);
        return true;
    }
    bool Snapshot(const gz::msgs::Boolean &, gz::msgs::Double_V &response) {
        std::lock_guard<std::mutex> lock(mutex);
        if (physics_capture) { std::fflush(physics_capture); }
        for (double v : snapshot) response.add_data(v);
        return true;
    }
    bool NativeSnapshot(const gz::msgs::Boolean &, gz::msgs::Double_V &response) {
        std::lock_guard<std::mutex> lock(mutex);
        for (double v : native_snapshot) response.add_data(v);
        return true;
    }
    bool NativeAction(const gz::msgs::Double_V &request, gz::msgs::Boolean &response) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!native_torque || request.data_size() != 2 || !std::isfinite(request.data(1))) {
            response.set_data(false); return true;
        }
        if (request.data(0) == double(requested + 1) && request.data(1) > double(expected_native_sample)) {
            expected_native_sample = uint64_t(request.data(1));
            first_native_sample = 0;
            active = true;
            ++requested;
        } else if (request.data(0) != double(requested) || request.data(1) != double(expected_native_sample)) {
            response.set_data(false); return true;
        }
        // Never block this transport service waiting for another transport
        // callback. The bridge polls until PreUpdate has committed the command.
        response.set_data(committed >= requested && first_native_sample == expected_native_sample);
        return true;
    }
    void PostUpdate(const gz::sim::UpdateInfo &info, const gz::sim::EntityComponentManager &ecm) override {
        if (info.paused) return;
        std::lock_guard<std::mutex> lock(mutex);
        snapshot.fill(NAN);
        snapshot[0] = 2;
        snapshot[1] = 0;
        snapshot[2] = std::chrono::duration_cast<std::chrono::microseconds>(info.simTime).count();
        snapshot[3] = committed;
        for (int i = 0; i < 4; ++i) snapshot[4 + i] = applied_motors[i];
        native_snapshot[0] = snapshot[2]; native_snapshot[1] = committed;
        native_snapshot[2] = committed_native_sample; native_snapshot[3] = first_native_sample;
        for (int i = 0; i < 4; ++i) native_snapshot[4 + i] = applied_allocated[i];
        const auto pose = base.WorldPose(ecm);
        const auto velocity_world = base.WorldLinearVelocity(ecm);
        const auto rates_world = base.WorldAngularVelocity(ecm);
        if (!pose || !velocity_world || !rates_world) return;
        snapshot[12] = pose->Pos().Y();
        snapshot[13] = pose->Pos().X();
        snapshot[14] = -pose->Pos().Z();
        const gz::math::Quaterniond enu_to_ned(0, std::sqrt(0.5), std::sqrt(0.5), 0);
        const gz::math::Quaterniond flu_to_frd(0, 1, 0, 0);
        auto q = enu_to_ned * pose->Rot() * flu_to_frd.Inverse();
        q.Normalize();
        snapshot[15] = q.W(); snapshot[16] = q.X();
        snapshot[17] = q.Y(); snapshot[18] = q.Z();
        snapshot[19] = velocity_world->Y();
        snapshot[20] = velocity_world->X();
        snapshot[21] = -velocity_world->Z();
        const auto rates_body = pose->Rot().RotateVectorReverse(*rates_world);
        snapshot[22] = rates_body.X();
        snapshot[23] = -rates_body.Y();
        snapshot[24] = -rates_body.Z();
        for (int i = 0; i < 4; ++i) {
            const auto *velocity = ecm.Component<gz::sim::components::JointVelocity>(rotors[i]);
            if (!velocity || velocity->Data().empty()) return;
            snapshot[8 + i] = std::abs(velocity->Data()[0]) * slowdown / max_speed;
        }
        for (double v : snapshot) if (!std::isfinite(v)) return;
        snapshot[1] = 1;
        notify_sync();
        if (physics_capture && snapshot[2] > captured_time) {
            std::fwrite(snapshot.data(), sizeof(double), snapshot.size(), physics_capture);
            captured_time = snapshot[2];
        }
    }
    bool Action(const gz::msgs::Double_V &request, gz::msgs::Boolean &response) {
        std::unique_lock<std::mutex> lock(mutex);
        if (native_torque || request.data_size() != 5 || request.data(0) != double(requested + 1)) {
            response.set_data(false); return true;
        }
        for (int i = 0; i < 4; ++i) {
            if (!std::isfinite(request.data(i + 1)) || request.data(i + 1) < 0 || request.data(i + 1) > 1) {
                response.set_data(false); return true;
            }
        }
        // Native PID audit: STEP is only a simulation clock request. Never replace
        // asynchronously allocated native motor outputs with Python samples.
        if (native_baseline) {
            active = true;
            ++requested;
            response.set_data(true);
            return true;
        }
        for (int i = 0; i < 4; ++i) motors[i] = request.data(i + 1);
        active = true;
        const auto generation = ++requested;
        response.set_data(applied.wait_for(lock, std::chrono::seconds(3),
            [&] { return committed >= generation; }));
        return true;
    }
    void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override {
        std::lock_guard<std::mutex> lock(mutex);
        // Bootstrap only: hold the fresh vehicle in the air while PX4 sensors
        // initialize. A new simulator process resets rotor filters every episode.
        if (!active && !native_bootstrap) {
            model.SetWorldPoseCmd(ecm, gz::math::Pose3d(0, 0, spawn_height, 0, 0, 0));
            base.SetLinearVelocity(ecm, gz::math::Vector3d::Zero);
            base.SetAngularVelocity(ecm, gz::math::Vector3d::Zero);
        } else {
            // Gazebo clears these commands to zero but retains the components.
            // Remove them when bootstrap ends, otherwise velocity overrides can
            // keep suppressing gravity/torques after the first learned action.
            ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(base.Entity());
            ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(base.Entity());
        }
        gz::msgs::Actuators command;
        for (double m : motors) command.add_velocity(m * max_speed);
        ecm.SetComponentData<gz::sim::components::Actuators>(model.Entity(), command);
        if (closed_loop_sync && !info.paused) {sync_applied=sync_received;sync_applied_valid=sync_received_valid;}
        applied_motors = motors;
        applied_allocated = allocated;
        committed_native_sample = native_sample;
        if (!native_torque || !requested || native_sample >= expected_native_sample) {
            if (native_torque && committed < requested) first_native_sample = native_sample;
            committed = requested;
        }
        applied.notify_all();
    }
};
GZ_ADD_PLUGIN(RateTrainingSystem, gz::sim::System,
    gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate, gz::sim::ISystemConfigurePriority,
    gz::sim::ISystemPostUpdate)
