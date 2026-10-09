#pragma once
#include <cmath>
#include <cstdint>
// NED positive down. All altitudes must share the same MSL datum.
// Only test-fixture world metadata and FC origin are used, never live truth.
inline bool bootstrap_local_z(double world_elevation_msl, double height_above_world,
 double local_ref_alt_msl, bool z_global, uint64_t ref_timestamp, float &out) {
 if(!z_global || ref_timestamp==0 || !std::isfinite(world_elevation_msl)
    || !std::isfinite(height_above_world) || !std::isfinite(local_ref_alt_msl)) return false;
 const double goal=local_ref_alt_msl-(world_elevation_msl+height_above_world);
 if(!std::isfinite(goal) || std::abs(goal)>1000000.)return false;
 out=static_cast<float>(goal);return true;
}
