// Sensor-side bias random walk, before PX4's sensor processing. The stock
// Gazebo IMU already supplies white noise; do not add that noise a second time.
#include <gz/sim/System.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/imu.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <random>

class ImuRandomWalkSystem : public gz::sim::System,
                            public gz::sim::ISystemConfigure {
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    sigma = sdf->Get<double>("bias_random_walk");
    const auto topic = sdf->Get<std::string>("input_topic");
    const char *seed = std::getenv("RATE_RL_NOISE_SEED");
    rng.seed(seed ? std::strtoul(seed, nullptr, 10) : 0);
    publisher = node.Advertise<gz::msgs::IMU>(topic + "/rl_noisy");
    node.Advertise("/rate_training/noise", &ImuRandomWalkSystem::Snapshot, this);
    node.Subscribe(topic, &ImuRandomWalkSystem::Imu, this);
  }

 private:
  void Imu(const gz::msgs::IMU &input) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto &stamp = input.header().stamp();
    const int64_t now = int64_t(stamp.sec()) * 1000000 + stamp.nsec() / 1000;
    if (now == previous && initialized) return; // No random draw while paused.
    if (!initialized || now < previous) {
      bias.fill(0);
    } else {
      const double scale = sigma * std::sqrt((now - previous) * 1e-6);
      for (double &b : bias) b += scale * normal(rng);
    }
    initialized = true;
    previous = now;
    gz::msgs::IMU output = input;
    const auto &raw = input.angular_velocity();
    const std::array<double, 3> values{raw.x(), raw.y(), raw.z()};
    auto *noisy = output.mutable_angular_velocity();
    noisy->set_x(values[0] + bias[0]);
    noisy->set_y(values[1] + bias[1]);
    noisy->set_z(values[2] + bias[2]);
    snapshot.Clear();
    snapshot.add_data(now);
    for (double b : bias) snapshot.add_data(b);
    for (double v : values) snapshot.add_data(v);
    for (unsigned i = 0; i < 3; ++i) snapshot.add_data(values[i] + bias[i]);
    publisher.Publish(output);
  }
  bool Snapshot(const gz::msgs::Boolean &, gz::msgs::Double_V &output) {
    std::lock_guard<std::mutex> lock(mutex);
    output = snapshot;
    return initialized;
  }
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher;
  std::mutex mutex;
  std::mt19937 rng;
  std::normal_distribution<double> normal{0, 1};
  std::array<double, 3> bias{};
  gz::msgs::Double_V snapshot;
  double sigma{1e-4}; // (rad/s)/sqrt(s); engineering starting value.
  int64_t previous{0};
  bool initialized{false};
};

GZ_ADD_PLUGIN(ImuRandomWalkSystem, gz::sim::System, ImuRandomWalkSystem::ISystemConfigure)
