/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <pcl/common/transforms.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <numeric>
#include <unordered_map>
#include <sstream>

// ========================= helpers (anonymous namespace) =========================
namespace
{
constexpr double kPi = 3.14159265358979323846;

double normalize_angle(double a)
{
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

double percentile_from_sorted(const std::vector<double> &vals, double q01)
{
  if (vals.empty()) return 0.0;
  const double q = std::clamp(q01, 0.0, 1.0);
  const auto idx = static_cast<std::size_t>(q * static_cast<double>(vals.size() - 1));
  return vals[idx];
}

// Task1 shape occupancy helpers (5×5 grid, cell size 40 mm).
bool nought_cell_occupied(int ix, int iy) { return std::abs(ix) == 2 || std::abs(iy) == 2; }
bool cross_cell_occupied(int ix, int iy)   { return ix == 0 || iy == 0; }

double point_to_rect_distance(double px, double py, double cx, double cy, double hx, double hy)
{
  const double dx = std::max(std::abs(px - cx) - hx, 0.0);
  const double dy = std::max(std::abs(py - cy) - hy, 0.0);
  return std::hypot(dx, dy);
}

double template_fit_cost(
  const PointC &surface, const geometry_msgs::msg::Point &center,
  bool is_nought, bool is_cross, double yaw, double cell)
{
  std::vector<double> dists;
  dists.reserve(surface.size());
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double half = 0.5 * cell;

  for (const auto &p : surface.points) {
    const double dx = p.x - center.x, dy = p.y - center.y;
    const double lx = c * dx + s * dy, ly = -s * dx + c * dy;
    double best = std::numeric_limits<double>::infinity();
    for (int ix = -2; ix <= 2; ++ix) {
      for (int iy = -2; iy <= 2; ++iy) {
        const bool occ = (is_nought && nought_cell_occupied(ix, iy)) ||
                         (is_cross  && cross_cell_occupied(ix, iy));
        if (!occ) continue;
        best = std::min(best, point_to_rect_distance(
          lx, ly, ix * cell, iy * cell, half, half));
      }
    }
    if (std::isfinite(best)) dists.push_back(best);
  }
  if (dists.empty()) return std::numeric_limits<double>::infinity();
  std::sort(dists.begin(), dists.end());
  return percentile_from_sorted(dists, 0.78);
}

Eigen::Affine3f affine_from_tf(const geometry_msgs::msg::Transform &t)
{
  Eigen::Translation3f tr(
    static_cast<float>(t.translation.x),
    static_cast<float>(t.translation.y),
    static_cast<float>(t.translation.z));
  Eigen::Quaternionf q(
    static_cast<float>(t.rotation.w),
    static_cast<float>(t.rotation.x),
    static_cast<float>(t.rotation.y),
    static_cast<float>(t.rotation.z));
  return Eigen::Affine3f(tr * q);
}








enum class Task2ShapeLabel
{
  UNKNOWN = 0,
  NOUGHT = 1,
  CROSS = 2
};

struct Task2ShapeResult
{
  Task2ShapeLabel label = Task2ShapeLabel::UNKNOWN;
  double best_yaw = 0.0;
  double nought_cost = std::numeric_limits<double>::infinity();
  double cross_cost = std::numeric_limits<double>::infinity();
  std::size_t n_surface = 0;
};

struct Task2ObservedObject
{
  bool ok = false;
  geometry_msgs::msg::Point refined_center;
  PointC coarse_roi;
  PointC top_band;
  PointC surface_roi;
  Task2ShapeResult cls;
};

const char *task2_label_str(Task2ShapeLabel label)
{
  switch (label) {
    case Task2ShapeLabel::NOUGHT: return "nought";
    case Task2ShapeLabel::CROSS:  return "cross";
    default: return "unknown";
  }
}

void extract_task2_roi_xy_only(
  const PointC &cloud,
  const geometry_msgs::msg::Point &center,
  double radius_xy,
  PointC &roi)
{
  roi.clear();
  roi.header = cloud.header;
  const double r2 = radius_xy * radius_xy;

  for (const auto &p : cloud.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    const double dx = p.x - center.x;
    const double dy = p.y - center.y;
    if (dx * dx + dy * dy > r2) continue;
    roi.push_back(p);
  }
}

bool task2_find_refined_center_from_roi(
  const PointC &roi,
  geometry_msgs::msg::Point &refined_center,
  PointC &top_band,
  PointC &surface_roi)
{
  top_band.clear();
  surface_roi.clear();
  top_band.header = roi.header;
  surface_roi.header = roi.header;

  if (roi.size() < 20U) return false;

  std::vector<double> zs;
  zs.reserve(roi.size());
  for (const auto &p : roi.points) zs.push_back(p.z);
  std::sort(zs.begin(), zs.end());

  const double z_top = percentile_from_sorted(zs, 0.97);
  const double z_band_low = z_top - 0.025;
  const double z_band_high = z_top + 0.010;

  for (const auto &p : roi.points) {
    if (p.z >= z_band_low && p.z <= z_band_high) {
      top_band.push_back(p);
    }
  }

  if (top_band.size() < 15U) return false;

  double sx = 0.0, sy = 0.0, sz = 0.0;
  for (const auto &p : top_band.points) {
    sx += p.x;
    sy += p.y;
    sz += p.z;
  }

  refined_center.x = sx / static_cast<double>(top_band.size());
  refined_center.y = sy / static_cast<double>(top_band.size());
  refined_center.z = sz / static_cast<double>(top_band.size());

  constexpr double kSurfaceRadius = 0.075;
  const double r2 = kSurfaceRadius * kSurfaceRadius;

  for (const auto &p : roi.points) {
    const double dx = p.x - refined_center.x;
    const double dy = p.y - refined_center.y;
    if (dx * dx + dy * dy > r2) continue;
    if (p.z < z_band_low) continue;
    surface_roi.push_back(p);
  }

  if (surface_roi.size() < 15U) {
    surface_roi = top_band;
  }

  return !surface_roi.empty();
}

Task2ShapeResult classify_task2_shape_from_surface(
  const PointC &surface,
  const geometry_msgs::msg::Point &center)
{
  Task2ShapeResult result;
  result.n_surface = surface.size();
  if (surface.size() < 10U) return result;

  constexpr double kCell = 0.040;
  const int n_samples = 90;

  double best_nought_cost = std::numeric_limits<double>::infinity();
  double best_cross_cost  = std::numeric_limits<double>::infinity();
  double best_nought_yaw  = 0.0;
  double best_cross_yaw   = 0.0;

  for (int i = 0; i < n_samples; ++i) {
    const double yaw = (0.5 * kPi) * static_cast<double>(i) / static_cast<double>(n_samples);

    const double cn = template_fit_cost(surface, center, true, false, yaw, kCell);
    if (cn < best_nought_cost) {
      best_nought_cost = cn;
      best_nought_yaw = yaw;
    }

    const double cc = template_fit_cost(surface, center, false, true, yaw, kCell);
    if (cc < best_cross_cost) {
      best_cross_cost = cc;
      best_cross_yaw = yaw;
    }
  }

  for (double d : {0.0, 0.015, -0.015, 0.03, -0.03, 0.06, -0.06}) {
    const double yn = normalize_angle(best_nought_yaw + d);
    const double yc = normalize_angle(best_cross_yaw + d);

    const double cn = template_fit_cost(surface, center, true, false, yn, kCell);
    if (cn < best_nought_cost) {
      best_nought_cost = cn;
      best_nought_yaw = yn;
    }

    const double cc = template_fit_cost(surface, center, false, true, yc, kCell);
    if (cc < best_cross_cost) {
      best_cross_cost = cc;
      best_cross_yaw = yc;
    }
  }

  result.nought_cost = best_nought_cost;
  result.cross_cost  = best_cross_cost;

  if (best_nought_cost <= best_cross_cost) {
    result.label = Task2ShapeLabel::NOUGHT;
    result.best_yaw = best_nought_yaw;
  } else {
    result.label = Task2ShapeLabel::CROSS;
    result.best_yaw = best_cross_yaw;
  }

  return result;
}







struct Task3Object
{
  geometry_msgs::msg::Point center;
  PointC points;
  Task2ShapeLabel label = Task2ShapeLabel::UNKNOWN;
  int observations = 0;
};

struct Task3Obstacle
{
  double min_x = 0, max_x = 0;
  double min_y = 0, max_y = 0;
  double min_z = 0, max_z = 0;
  int observations = 0;
};

bool is_black_point(const PointT &p)
{
  return p.r < 45 && p.g < 45 && p.b < 45;
}

bool is_brown_basket_point(const PointT &p)
{
  return p.r > 80 && p.r < 170 &&
         p.g > 25 && p.g < 95 &&
         p.b > 25 && p.b < 95 &&
         p.r > p.g * 1.5;
}

bool is_grass_point(const PointT &p)
{
  return p.g > p.r * 1.25 && p.g > p.b * 1.25;
}

bool is_shape_candidate_point(const PointT &p)
{
  if (is_black_point(p)) return false;
  if (is_brown_basket_point(p)) return false;
  if (is_grass_point(p)) return false;
  return true;
}

void voxel_filter_xy(
  const PointC &in,
  PointC &out,
  double voxel = 0.006)
{
  out.clear();
  out.header = in.header;

  std::unordered_map<long long, PointT> grid;
  grid.reserve(in.size());

  for (const auto &p : in.points) {
    const long long ix = static_cast<long long>(std::floor(p.x / voxel));
    const long long iy = static_cast<long long>(std::floor(p.y / voxel));
    const long long iz = static_cast<long long>(std::floor(p.z / voxel));
    const long long key = (ix * 73856093LL) ^ (iy * 19349663LL) ^ (iz * 83492791LL);
    if (grid.find(key) == grid.end()) grid[key] = p;
  }

  for (const auto &kv : grid) out.push_back(kv.second);
}

void simple_cluster_xy(
  const PointC &cloud,
  double dist,
  std::vector<PointC> &clusters)
{
  clusters.clear();
  const int n = static_cast<int>(cloud.size());
  if (n == 0) return;

  std::vector<int> visited(n, 0);
  const double d2 = dist * dist;

  for (int i = 0; i < n; ++i) {
    if (visited[i]) continue;

    PointC cluster;
    cluster.header = cloud.header;

    std::vector<int> stack;
    stack.push_back(i);
    visited[i] = 1;

    while (!stack.empty()) {
      const int idx = stack.back();
      stack.pop_back();
      cluster.push_back(cloud.points[idx]);

      const auto &a = cloud.points[idx];
      for (int j = 0; j < n; ++j) {
        if (visited[j]) continue;
        const auto &b = cloud.points[j];
        const double dx = a.x - b.x;
        const double dy = a.y - b.y;
        if (dx * dx + dy * dy <= d2) {
          visited[j] = 1;
          stack.push_back(j);
        }
      }
    }

    if (cluster.size() >= 12U) clusters.push_back(cluster);
  }
}

geometry_msgs::msg::Point centroid_of_cloud(const PointC &cloud)
{
  geometry_msgs::msg::Point c;
  if (cloud.empty()) return c;

  double sx = 0, sy = 0, sz = 0;
  for (const auto &p : cloud.points) {
    sx += p.x;
    sy += p.y;
    sz += p.z;
  }

  c.x = sx / static_cast<double>(cloud.size());
  c.y = sy / static_cast<double>(cloud.size());
  c.z = sz / static_cast<double>(cloud.size());
  return c;
}

Task2ShapeResult classify_task3_cluster(const PointC &cluster)
{
  PointC surface = cluster;
  geometry_msgs::msg::Point c = centroid_of_cloud(surface);
  return classify_task2_shape_from_surface(surface, c);
}

void merge_task3_object(
  std::vector<Task3Object> &objects,
  const Task3Object &candidate)
{
  for (auto &obj : objects) {
    if (std::hypot(obj.center.x - candidate.center.x,
                   obj.center.y - candidate.center.y) < 0.075) {
      obj.points += candidate.points;
      obj.center = centroid_of_cloud(obj.points);
      obj.label = classify_task3_cluster(obj.points).label;
      obj.observations++;
      return;
    }
  }

  objects.push_back(candidate);
}

















void cloud_bounds(
  const PointC &cloud,
  double &min_x, double &max_x,
  double &min_y, double &max_y,
  double &min_z, double &max_z)
{
  min_x = min_y = min_z =  std::numeric_limits<double>::infinity();
  max_x = max_y = max_z = -std::numeric_limits<double>::infinity();

  for (const auto &p : cloud.points) {
    min_x = std::min(min_x, static_cast<double>(p.x));
    max_x = std::max(max_x, static_cast<double>(p.x));
    min_y = std::min(min_y, static_cast<double>(p.y));
    max_y = std::max(max_y, static_cast<double>(p.y));
    min_z = std::min(min_z, static_cast<double>(p.z));
    max_z = std::max(max_z, static_cast<double>(p.z));
  }
}

void merge_task3_obstacle(
  std::vector<Task3Obstacle> &obstacles,
  const Task3Obstacle &candidate)
{
  const double cx = 0.5 * (candidate.min_x + candidate.max_x);
  const double cy = 0.5 * (candidate.min_y + candidate.max_y);

  for (auto &o : obstacles) {
    const double ox = 0.5 * (o.min_x + o.max_x);
    const double oy = 0.5 * (o.min_y + o.max_y);
    if (std::hypot(cx - ox, cy - oy) < 0.13) {
      o.min_x = std::min(o.min_x, candidate.min_x);
      o.max_x = std::max(o.max_x, candidate.max_x);
      o.min_y = std::min(o.min_y, candidate.min_y);
      o.max_y = std::max(o.max_y, candidate.max_y);
      o.min_z = std::min(o.min_z, candidate.min_z);
      o.max_z = std::max(o.max_z, candidate.max_z);
      o.observations++;
      return;
    }
  }

  obstacles.push_back(candidate);
}



Task2ShapeResult classify_task3_cluster_multiscale(
  const PointC &cluster,
  geometry_msgs::msg::Point *best_center_out = nullptr,
  double *best_cell_out = nullptr)
{
  Task2ShapeResult best_result;
  best_result.label = Task2ShapeLabel::UNKNOWN;

  if (cluster.size() < 10U) return best_result;

  double min_x, max_x, min_y, max_y, min_z, max_z;
  cloud_bounds(cluster, min_x, max_x, min_y, max_y, min_z, max_z);

  geometry_msgs::msg::Point center;
  center.x = 0.5 * (min_x + max_x);
  center.y = 0.5 * (min_y + max_y);
  center.z = 0.5 * (min_z + max_z);

  const std::vector<double> cells = {0.020, 0.030, 0.040};

  double best_score = std::numeric_limits<double>::infinity();
  double best_cell = 0.040;
  double best_yaw = 0.0;
  Task2ShapeLabel best_label = Task2ShapeLabel::UNKNOWN;
  double best_nought_cost = std::numeric_limits<double>::infinity();
  double best_cross_cost = std::numeric_limits<double>::infinity();

  for (double cell : cells) {
    for (int i = 0; i < 90; ++i) {
      const double yaw = (0.5 * kPi) * static_cast<double>(i) / 90.0;

      const double n_cost = template_fit_cost(cluster, center, true, false, yaw, cell);
      const double c_cost = template_fit_cost(cluster, center, false, true, yaw, cell);

      const double n_score = n_cost + 0.002 * std::abs(cell - 0.030);
      const double c_score = c_cost + 0.002 * std::abs(cell - 0.030);

      if (n_score < best_score) {
        best_score = n_score;
        best_cell = cell;
        best_yaw = yaw;
        best_label = Task2ShapeLabel::NOUGHT;
        best_nought_cost = n_cost;
        best_cross_cost = c_cost;
      }

      if (c_score < best_score) {
        best_score = c_score;
        best_cell = cell;
        best_yaw = yaw;
        best_label = Task2ShapeLabel::CROSS;
        best_nought_cost = n_cost;
        best_cross_cost = c_cost;
      }
    }
  }

  best_result.label = best_label;
  best_result.best_yaw = best_yaw;
  best_result.nought_cost = best_nought_cost;
  best_result.cross_cost = best_cross_cost;
  best_result.n_surface = cluster.size();

  if (best_center_out) *best_center_out = center;
  if (best_cell_out) *best_cell_out = best_cell;

  return best_result;
}




















}  // namespace

// ========================= Constructor =========================
cw2::cw2(const rclcpp::Node::SharedPtr &node)
: node_(node),
  tf_buffer_(node->get_clock()),
  tf_listener_(tf_buffer_),
  g_cloud_ptr(new PointC)
{
  task_service_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  t1_service_ = node_->create_service<cw2_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw2::t1_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, task_service_callback_group_);
  t2_service_ = node_->create_service<cw2_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw2::t2_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, task_service_callback_group_);
  t3_service_ = node_->create_service<cw2_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw2::t3_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, task_service_callback_group_);

  // --- ROS parameters (general) ---
  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ = node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

  cloud_max_age_sec_   = node_->declare_parameter<double>("cloud_max_age_sec", cloud_max_age_sec_);
  cloud_wait_timeout_sec_ = node_->declare_parameter<double>("cloud_wait_timeout_sec", cloud_wait_timeout_sec_);
  spawn_settle_ms_     = node_->declare_parameter<int>("spawn_settle_ms", spawn_settle_ms_);
  gripper_open_width_  = node_->declare_parameter<double>("gripper_open_width", gripper_open_width_);
  gripper_max_width_   = node_->declare_parameter<double>("gripper_max_width", gripper_max_width_);
  gripper_min_width_   = node_->declare_parameter<double>("gripper_min_width", gripper_min_width_);
  blocked_close_width_tolerance_ = node_->declare_parameter<double>(
    "blocked_close_width_tolerance", blocked_close_width_tolerance_);

  // --- ROS parameters (Task 1) ---
  task1_roi_radius_xy_       = node_->declare_parameter<double>("task1_roi_radius_xy", task1_roi_radius_xy_);
  task1_roi_half_height_     = node_->declare_parameter<double>("task1_roi_half_height", task1_roi_half_height_);
  task1_surface_percentile_  = node_->declare_parameter<double>("task1_surface_percentile", task1_surface_percentile_);
  task1_ground_reject_height_= node_->declare_parameter<double>("task1_ground_reject_height", task1_ground_reject_height_);
  task1_gripper_closing_margin_ = node_->declare_parameter<double>("task1_gripper_closing_margin", task1_gripper_closing_margin_);
  task1_candidate_strip_half_depth_ = node_->declare_parameter<double>("task1_candidate_strip_half_depth", task1_candidate_strip_half_depth_);
  task1_contact_side_threshold_ = node_->declare_parameter<double>("task1_contact_side_threshold", task1_contact_side_threshold_);
  task1_min_contact_points_per_side_ = node_->declare_parameter<int>("task1_min_contact_points_per_side", task1_min_contact_points_per_side_);
  task1_max_candidates_to_try_ = node_->declare_parameter<int>("task1_max_candidates_to_try", task1_max_candidates_to_try_);
  task1_n_yaw_samples_       = node_->declare_parameter<int>("task1_n_yaw_samples", task1_n_yaw_samples_);
  task1_pregrasp_offset_z_   = node_->declare_parameter<double>("task1_pregrasp_offset_z", task1_pregrasp_offset_z_);
  task1_grasp_offset_z_      = node_->declare_parameter<double>("task1_grasp_offset_z", task1_grasp_offset_z_);
  task1_lift_offset_z_       = node_->declare_parameter<double>("task1_lift_offset_z", task1_lift_offset_z_);
  task1_place_offset_z_      = node_->declare_parameter<double>("task1_place_offset_z", task1_place_offset_z_);
  task1_release_min_extra_z_ = node_->declare_parameter<double>("task1_release_min_extra_z", task1_release_min_extra_z_);
  task1_release_max_extra_z_ = node_->declare_parameter<double>("task1_release_max_extra_z", task1_release_max_extra_z_);
  task1_release_object_height_scale_ = node_->declare_parameter<double>("task1_release_object_height_scale", task1_release_object_height_scale_);
  task1_basket_approach_min_z_ = node_->declare_parameter<double>("task1_basket_approach_min_z", task1_basket_approach_min_z_);
  task1_transit_raise_above_lift_ = node_->declare_parameter<double>("task1_transit_raise_above_lift", task1_transit_raise_above_lift_);
  task1_transit_xy_vel_scale_ = node_->declare_parameter<double>("task1_transit_xy_vel_scale", task1_transit_xy_vel_scale_);
  task1_transit_xy_acc_scale_ = node_->declare_parameter<double>("task1_transit_xy_acc_scale", task1_transit_xy_acc_scale_);
  task1_place_descend_vel_scale_ = node_->declare_parameter<double>("task1_place_descend_vel_scale", task1_place_descend_vel_scale_);
  task1_place_descend_acc_scale_ = node_->declare_parameter<double>("task1_place_descend_acc_scale", task1_place_descend_acc_scale_);
  task1_ee_yaw_offset_rad_   = node_->declare_parameter<double>("task1_ee_yaw_offset_rad", task1_ee_yaw_offset_rad_);
  task1_descend_velocity_scale_ = node_->declare_parameter<double>("task1_descend_velocity_scale", task1_descend_velocity_scale_);
  task1_descend_acceleration_scale_ = node_->declare_parameter<double>("task1_descend_acceleration_scale", task1_descend_acceleration_scale_);
  task1_approach_last_delta_z_ = node_->declare_parameter<double>("task1_approach_last_delta_z", task1_approach_last_delta_z_);
  task1_grasp_settle_ms_     = node_->declare_parameter<int>("task1_grasp_settle_ms", task1_grasp_settle_ms_);
  task1_grasp_backoff_z_     = node_->declare_parameter<double>("task1_grasp_backoff_z", task1_grasp_backoff_z_);

  // --- Point cloud subscription ---
  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pc_opts;
  pc_opts.callback_group = pointcloud_callback_group_;

  rclcpp::QoS pc_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_)
    pc_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();

  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_, pc_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1), pc_opts);

  // --- MoveIt groups ---
  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");
  setup_arm_hand(*arm_group_, *hand_group_);

  RCLCPP_INFO(node_->get_logger(),
    "cw2_team_8 ready. Cloud='%s' (%s) pose_ref=%s planning=%s",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data",
    arm_group_->getPoseReferenceFrame().c_str(),
    arm_group_->getPlanningFrame().c_str());
}

// ======================== Common utilities ========================

void cw2::setup_arm_hand(
  moveit::planning_interface::MoveGroupInterface &arm,
  moveit::planning_interface::MoveGroupInterface &hand) const
{
  arm.setPoseReferenceFrame("panda_link0");
  arm.setPlanningTime(12.0);
  arm.setNumPlanningAttempts(8);
  arm.setMaxVelocityScalingFactor(0.35);
  arm.setMaxAccelerationScalingFactor(0.35);
  arm.setGoalTolerance(0.002);
  arm.setPlannerId("RRTConnect");

  hand.setPlanningTime(5.0);
  hand.setNumPlanningAttempts(5);
  hand.setMaxVelocityScalingFactor(0.5);
  hand.setMaxAccelerationScalingFactor(0.5);
}

// -- Point cloud --

void cw2::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  pcl::PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*msg, pcl_cloud);
  PointCPtr latest(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest);

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  g_input_pc_frame_id_ = msg->header.frame_id;
  g_cloud_ptr = std::move(latest);
  ++g_cloud_sequence_;
  g_cloud_receive_steady_ = std::chrono::steady_clock::now();
}

bool cw2::wait_for_cloud(
  std::chrono::milliseconds timeout,
  std::uint64_t min_seq,
  std::uint64_t *out_seq) const
{
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 < timeout) {
    std::uint64_t seq = 0;
    std::chrono::steady_clock::time_point tp = std::chrono::steady_clock::time_point::min();
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      seq = g_cloud_sequence_;
      tp  = g_cloud_receive_steady_;
    }
    if (seq > min_seq && tp != std::chrono::steady_clock::time_point::min()) {
      const auto age = std::chrono::steady_clock::now() - tp;
      if (age > std::chrono::duration<double>(std::max(0.05, cloud_max_age_sec_))) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      if (out_seq) *out_seq = seq;
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

bool cw2::cloud_recently_valid(std::chrono::seconds max_age) const
{
  std::lock_guard<std::mutex> lk(cloud_mutex_);
  if (!g_cloud_ptr || g_cloud_ptr->empty()) return false;
  if (g_cloud_receive_steady_ == std::chrono::steady_clock::time_point::min()) return false;
  return (std::chrono::steady_clock::now() - g_cloud_receive_steady_) <= max_age;
}

bool cw2::get_cloud_snapshot(PointCPtr &cloud, std::string &frame_id, std::uint64_t &sequence) const
{
  std::lock_guard<std::mutex> lk(cloud_mutex_);
  if (!g_cloud_ptr || g_cloud_ptr->empty()) return false;
  cloud.reset(new PointC(*g_cloud_ptr));
  frame_id = g_input_pc_frame_id_;
  sequence = g_cloud_sequence_;
  return true;
}

// -- Transform --

bool cw2::transform_point_to_link0(
  const geometry_msgs::msg::PointStamped &in,
  geometry_msgs::msg::PointStamped &out) const
{
  if (in.header.frame_id.empty() || in.header.frame_id == "panda_link0") {
    out = in;
    out.header.frame_id = "panda_link0";
    out.header.stamp = node_->get_clock()->now();
    return true;
  }
  try {
    auto tf = tf_buffer_.lookupTransform("panda_link0", in.header.frame_id,
                                         tf2::TimePointZero, tf2::durationFromSec(3.0));
    tf2::doTransform(in, out, tf);
    out.header.frame_id = "panda_link0";
    out.header.stamp = node_->get_clock()->now();
    return true;
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF point->panda_link0: %s", ex.what());
    return false;
  }
}

bool cw2::transform_cloud_to_frame(
  const PointC &cloud_in, const std::string &src, const std::string &dst, PointC &cloud_out) const
{
  if (src == dst) { cloud_out = cloud_in; return true; }
  try {
    auto tf = tf_buffer_.lookupTransform(dst, src, tf2::TimePointZero, tf2::durationFromSec(3.0));
    pcl::transformPointCloud(cloud_in, cloud_out, affine_from_tf(tf.transform));
    return true;
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF cloud %s->%s: %s", src.c_str(), dst.c_str(), ex.what());
    return false;
  }
}

// -- ROI extraction (general, parameter-driven) --

void cw2::extract_roi_points(
  const PointC &cloud, const geometry_msgs::msg::Point &center,
  double radius_xy, double half_h, double min_z, PointC &roi) const
{
  roi.clear();
  roi.header = cloud.header;
  const double r2 = radius_xy * radius_xy;
  for (const auto &p : cloud.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    const double dx = p.x - center.x, dy = p.y - center.y;
    if (dx * dx + dy * dy > r2) continue;
    if (std::abs(p.z - center.z) > half_h) continue;
    if (p.z < min_z) continue;
    roi.push_back(p);
  }
}

// -- Arm movement --

bool cw2::execute_plan(
  moveit::planning_interface::MoveGroupInterface &group,
  const moveit::planning_interface::MoveGroupInterface::Plan &plan) const
{
  return group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
}

bool cw2::move_arm_to_pose(const geometry_msgs::msg::Pose &target) const
{
  arm_group_->setStartStateToCurrentState();
  arm_group_->setPoseTarget(target);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  if (arm_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) return false;
  return execute_plan(*arm_group_, plan);
}

bool cw2::cartesian_move(const geometry_msgs::msg::Pose &target, double min_frac) const
{
  arm_group_->setStartStateToCurrentState();
  std::vector<geometry_msgs::msg::Pose> wp{target};
  moveit_msgs::msg::RobotTrajectory traj;
  const double frac = arm_group_->computeCartesianPath(wp, 0.002, 0.0, traj, false);
  if (frac < min_frac) return false;
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = traj;
  return arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
}

bool cw2::cartesian_follow_waypoints(
  const std::vector<geometry_msgs::msg::Pose> &waypoints,
  double min_frac, double vel, double acc, double step) const
{
  if (waypoints.empty()) return false;
  arm_group_->setMaxVelocityScalingFactor(vel);
  arm_group_->setMaxAccelerationScalingFactor(acc);
  arm_group_->setStartStateToCurrentState();
  moveit_msgs::msg::RobotTrajectory traj;
  const double frac = arm_group_->computeCartesianPath(waypoints, step, 0.0, traj, false);
  arm_group_->setMaxVelocityScalingFactor(0.35);
  arm_group_->setMaxAccelerationScalingFactor(0.35);
  if (frac < min_frac) return false;
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = traj;
  return arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
}

// -- Gripper --

bool cw2::set_gripper(double width_m) const
{
  const double w = std::clamp(width_m, 0.0, gripper_open_width_);
  hand_group_->setStartStateToCurrentState();
  hand_group_->setJointValueTarget("panda_finger_joint1", w / 2.0);
  hand_group_->setJointValueTarget("panda_finger_joint2", w / 2.0);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto code = hand_group_->plan(plan);
  if (code != moveit::core::MoveItErrorCode::SUCCESS) {
    const double w2 = std::clamp(w + 0.0008, 0.0, gripper_open_width_);
    hand_group_->setStartStateToCurrentState();
    hand_group_->setJointValueTarget("panda_finger_joint1", w2 / 2.0);
    hand_group_->setJointValueTarget("panda_finger_joint2", w2 / 2.0);
    code = hand_group_->plan(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) return false;
  }
  return execute_plan(*hand_group_, plan);
}

double cw2::current_gripper_width() const
{
  const auto v = hand_group_->getCurrentJointValues();
  if (v.size() >= 2U && std::isfinite(v[0]) && std::isfinite(v[1]))
    return std::max(0.0, v[0] + v[1]);
  return gripper_open_width_;
}

bool cw2::open_gripper() const
{
  return set_gripper(gripper_open_width_);
}

bool cw2::close_gripper_for_grasp(double target_width_m) const
{
  const double target = std::clamp(target_width_m, 0.0, gripper_open_width_);
  constexpr int kMaxRetries = 3;
  constexpr double kStep = 0.004;

  for (int r = 0; r < kMaxRetries; ++r) {
    const double try_w = std::clamp(target + r * kStep, 0.0, gripper_open_width_);
    if (set_gripper(try_w)) return true;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const double achieved = current_gripper_width();
    if (achieved <= std::min(gripper_open_width_ - 0.004, target + blocked_close_width_tolerance_)) {
      RCLCPP_INFO(node_->get_logger(),
        "Gripper blocked by object (achieved=%.4f target=%.4f) — treating as grasp", achieved, target);
      return true;
    }
  }
  return false;
}

// -- Collision scene helpers --

void cw2::add_collision_box(
  const std::string &id, double cx, double cy, double cz, double sx, double sy, double sz)
{
  moveit_msgs::msg::CollisionObject co;
  co.header.frame_id = "panda_link0";
  co.header.stamp = node_->get_clock()->now();
  co.id = id;
  co.operation = moveit_msgs::msg::CollisionObject::ADD;
  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions = {sx, sy, sz};
  geometry_msgs::msg::Pose p;
  p.position.x = cx;
  p.position.y = cy;
  p.position.z = cz;
  p.orientation.w = 1.0;
  co.primitives.push_back(box);
  co.primitive_poses.push_back(p);
  planning_scene_interface_.applyCollisionObject(co);
}

void cw2::add_ground_guard(const std::string &id)
{
  add_collision_box(id, 0.0, 0.0, -0.005, 2.0, 2.0, 0.01);
}

void cw2::remove_collision_objects(const std::vector<std::string> &ids)
{
  planning_scene_interface_.removeCollisionObjects(ids);
}

// -- Pose --

geometry_msgs::msg::Pose cw2::make_topdown_pose(double x, double y, double z, double yaw) const
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  tf2::Quaternion q;
  q.setRPY(0.0, kPi, yaw);
  pose.orientation = tf2::toMsg(q);
  return pose;
}

// ====================== Task 1 specifics =========================

double cw2::task1_arm_link8_yaw(double grasp_yaw) const
{
  return normalize_angle(grasp_yaw + task1_ee_yaw_offset_rad_);
}

void cw2::generate_task1_candidates(
  const PointC &roi,
  const geometry_msgs::msg::Point &obj,
  const std::string &shape_type,
  std::vector<Task1GraspCandidate> &out) const
{
  out.clear();
  const bool is_nought = shape_type.find("nought") != std::string::npos;
  const bool is_cross  = shape_type.find("cross")  != std::string::npos;
  if (!is_nought && !is_cross) return;

  // ===== OFFLINE: predefined grasp hypotheses (Lecture 7 model-based) =====
  struct Hyp { double dx, dy, yaw, close_w; };
  constexpr double kCell = 0.040;
  std::vector<Hyp> hyps;

  if (is_nought) {
    const double w = std::clamp(kCell - task1_gripper_closing_margin_, gripper_min_width_, gripper_max_width_);
    hyps.push_back({ 2*kCell,  0.0,       0.0,          w});
    hyps.push_back({-2*kCell,  0.0,       kPi,          w});
    hyps.push_back({ 0.0,      2*kCell,   0.5*kPi,      w});
    hyps.push_back({ 0.0,     -2*kCell,  -0.5*kPi,      w});
  } else {
    const double w = std::clamp(kCell - task1_gripper_closing_margin_, gripper_min_width_, gripper_max_width_);
    hyps.push_back({ 1.5*kCell, 0.0,  0.5*kPi, w});
    hyps.push_back({-1.5*kCell, 0.0,  0.5*kPi, w});
    hyps.push_back({0.0,  1.5*kCell,   0.0,     w});
    hyps.push_back({0.0, -1.5*kCell,   0.0,     w});
  }

  // ===== ONLINE: estimate object yaw via template fitting =====
  PointC surface;
  if (!roi.empty()) {
    std::vector<double> zs;
    zs.reserve(roi.size());
    for (const auto &p : roi.points) zs.push_back(p.z);
    std::sort(zs.begin(), zs.end());
    const double z_thr = percentile_from_sorted(zs, task1_surface_percentile_);
    surface.header = roi.header;
    for (const auto &p : roi.points)
      if (p.z >= z_thr) surface.push_back(p);
    if (surface.size() < 8U) surface = roi;
  }

  double object_yaw = 0.0;
  if (surface.size() >= 8U) {
    double best_cost = std::numeric_limits<double>::infinity();
    const int n = std::max(36, task1_n_yaw_samples_ * 3);
    for (int i = 0; i < n; ++i) {
      const double y = (0.5 * kPi) * i / static_cast<double>(n);
      const double c = template_fit_cost(surface, obj, is_nought, is_cross, y, kCell);
      if (c < best_cost) { best_cost = c; object_yaw = y; }
    }
    for (double d : {0.0, 0.02, -0.02, 0.04, -0.04, 0.08, -0.08}) {
      const double y = normalize_angle(object_yaw + d);
      const double c = template_fit_cost(surface, obj, is_nought, is_cross, y, kCell);
      if (c < best_cost) { best_cost = c; object_yaw = y; }
    }
  }

  // ===== Transform hypotheses to panda_link0 frame =====
  const double cy = std::cos(object_yaw), sy = std::sin(object_yaw);
  for (std::size_t i = 0; i < hyps.size(); ++i) {
    const auto &h = hyps[i];
    Task1GraspCandidate cand;
    cand.x = obj.x + cy * h.dx - sy * h.dy;
    cand.y = obj.y + sy * h.dx + cy * h.dy;
    cand.yaw = normalize_angle(h.yaw + object_yaw);
    cand.close_width = h.close_w;
    cand.score = static_cast<double>(hyps.size() - i);
    out.push_back(cand);
  }
}

// ========================= Task 1 callback =========================
void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  // --- Transform service points to panda_link0 ---
  geometry_msgs::msg::PointStamped obj_l0, goal_l0;
  if (!transform_point_to_link0(request->object_point, obj_l0) ||
      !transform_point_to_link0(request->goal_point, goal_l0))
  {
    RCLCPP_ERROR(node_->get_logger(), "Task1: TF to panda_link0 failed");
    return;
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task1 start: shape=%s object=(%.3f,%.3f,%.3f) goal=(%.3f,%.3f,%.3f)",
    request->shape_type.c_str(),
    obj_l0.point.x, obj_l0.point.y, obj_l0.point.z,
    goal_l0.point.x, goal_l0.point.y, goal_l0.point.z);

  // --- Wait for fresh point cloud ---
  if (spawn_settle_ms_ > 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(spawn_settle_ms_));

  std::uint64_t seq_before = 0;
  { std::lock_guard<std::mutex> lk(cloud_mutex_); seq_before = g_cloud_sequence_; }

  const auto wait_ms = std::chrono::milliseconds(
    static_cast<int>(std::max(1.0, cloud_wait_timeout_sec_) * 1000.0));
  if (!wait_for_cloud(wait_ms, seq_before, nullptr)) {
    if (cloud_recently_valid(std::chrono::seconds(3))) {
      RCLCPP_WARN(node_->get_logger(), "Task1: no new cloud seq; using latest");
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Task1: no fresh point cloud");
      return;
    }
  }

  PointCPtr cloud_cam;
  std::string cloud_frame;
  std::uint64_t cloud_seq = 0;
  if (!get_cloud_snapshot(cloud_cam, cloud_frame, cloud_seq) || !cloud_cam || cloud_cam->empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: empty cloud");
    return;
  }

  PointC cloud_l0;
  if (!transform_cloud_to_frame(*cloud_cam, cloud_frame, "panda_link0", cloud_l0)) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: cloud TF failed");
    return;
  }

  // --- Extract ROI & generate grasp candidates ---
  PointC roi;
  extract_roi_points(cloud_l0, obj_l0.point,
    task1_roi_radius_xy_, task1_roi_half_height_, task1_ground_reject_height_, roi);
  if (roi.size() < 40U) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: too few ROI points (%zu)", roi.size());
    return;
  }

  std::vector<Task1GraspCandidate> candidates;
  generate_task1_candidates(roi, obj_l0.point, request->shape_type, candidates);
  if (candidates.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: no grasp candidates");
    return;
  }

  // --- Grasp planning constants ---
  constexpr double kObjHeight = 0.040;
  const double grasp_z_nominal = obj_l0.point.z + 0.108;
  const std::string coll_obj_id = "task1_object";
  constexpr double kCollisionBoxH = 0.10;

  const int max_tries = std::max(1, task1_max_candidates_to_try_);
  bool grasp_ok = false;
  Task1GraspCandidate used{};
  double last_cx = 0, last_cy = 0, last_gz = 0, last_lz = 0;

  // --- Try each candidate ---
  for (int ti = 0; ti < std::min(max_tries, static_cast<int>(candidates.size())); ++ti) {
    const auto &c = candidates[static_cast<std::size_t>(ti)];
    const double yaw8 = task1_arm_link8_yaw(c.yaw);
    const double staging_yaw8 = task1_arm_link8_yaw(0.0);
    RCLCPP_INFO(node_->get_logger(),
      "Task1 try %d/%d: xy=(%.3f,%.3f) yaw=%.1fdeg close=%.4f",
      ti + 1, max_tries, c.x, c.y, c.yaw * 180.0 / kPi, c.close_width);

    if (!open_gripper()) { RCLCPP_WARN(node_->get_logger(), "Task1: open gripper failed"); continue; }

    // Height waypoints
    const double grasp_z = grasp_z_nominal;
    const double grasp_z_final = grasp_z + std::max(0.0, task1_grasp_backoff_z_);
    const double obj_top_z = obj_l0.point.z + kObjHeight;
    const double pre_floor = obj_top_z + 0.14;
    const double pre_z = std::max({
      obj_l0.point.z + task1_pregrasp_offset_z_,
      grasp_z + std::max(0.06, task1_approach_last_delta_z_ + 0.05),
      pre_floor});
    const double hover_z = std::min(0.48, std::max(pre_z + 0.06, pre_floor + 0.04));
    const double mid_z = std::min(hover_z - 0.012, std::max(pre_z + 0.028, grasp_z + 0.095));

    // Collision box around object for OMPL safety
    add_collision_box(coll_obj_id,
      obj_l0.point.x, obj_l0.point.y, obj_l0.point.z + kCollisionBoxH * 0.5,
      0.22, 0.22, kCollisionBoxH);

    // Move above object → candidate → align yaw
    auto obj_hover   = make_topdown_pose(obj_l0.point.x, obj_l0.point.y, hover_z, staging_yaw8);
    auto cand_hover  = make_topdown_pose(c.x, c.y, hover_z, staging_yaw8);
    auto cand_align  = make_topdown_pose(c.x, c.y, hover_z, yaw8);

    if (!move_arm_to_pose(obj_hover)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: move to object hover failed");
      remove_collision_objects({coll_obj_id}); continue;
    }
    if (std::hypot(c.x - obj_l0.point.x, c.y - obj_l0.point.y) > 0.010 &&
        !move_arm_to_pose(cand_hover)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: move to candidate hover failed");
      remove_collision_objects({coll_obj_id}); continue;
    }
    if (!move_arm_to_pose(cand_align)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: rotate above target failed");
      remove_collision_objects({coll_obj_id}); continue;
    }

    // Remove collision box, descend via Cartesian
    remove_collision_objects({coll_obj_id});

    std::vector<geometry_msgs::msg::Pose> drop;
    if (mid_z > pre_z + 0.012 && mid_z + 0.01 < hover_z)
      drop.push_back(make_topdown_pose(c.x, c.y, mid_z, yaw8));
    drop.push_back(make_topdown_pose(c.x, c.y, pre_z, yaw8));
    if (!cartesian_follow_waypoints(drop, 0.80, 0.11, 0.16, 0.001)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: drop to pre-grasp failed"); continue;
    }

    // Fine descent
    std::vector<geometry_msgs::msg::Pose> desc_hi{
      make_topdown_pose(c.x, c.y, grasp_z + task1_approach_last_delta_z_, yaw8),
      make_topdown_pose(c.x, c.y, grasp_z + 0.012, yaw8)};
    if (!cartesian_follow_waypoints(desc_hi, 0.80,
        task1_descend_velocity_scale_, task1_descend_acceleration_scale_, 0.001)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: descend upper failed"); continue;
    }
    std::vector<geometry_msgs::msg::Pose> desc_lo{
      make_topdown_pose(c.x, c.y, grasp_z + 0.012, yaw8),
      make_topdown_pose(c.x, c.y, grasp_z_final, yaw8)};
    if (!cartesian_follow_waypoints(desc_lo, 0.72, 0.035, 0.055, 0.0005)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: descend final failed"); continue;
    }

    // Grasp
    if (!close_gripper_for_grasp(c.close_width)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: close gripper failed"); continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, task1_grasp_settle_ms_)));

    // Lift
    const double lift_z = std::max(obj_l0.point.z + task1_lift_offset_z_, grasp_z + 0.055);
    if (!cartesian_move(make_topdown_pose(c.x, c.y, lift_z, yaw8), 0.72)) {
      RCLCPP_WARN(node_->get_logger(), "Task1: lift failed"); continue;
    }

    grasp_ok = true;
    used = c;
    last_cx = c.x; last_cy = c.y; last_gz = grasp_z_final; last_lz = lift_z;
    break;
  }

  if (!grasp_ok) {
    RCLCPP_ERROR(node_->get_logger(), "Task1: all grasp attempts failed");
    remove_collision_objects({coll_obj_id});
    return;
  }

  // --- Transit to basket ---
  const double extra_rel = std::clamp(
    task1_release_object_height_scale_ * kObjHeight,
    task1_release_min_extra_z_, task1_release_max_extra_z_);
  const double release_z = goal_l0.point.z + task1_place_offset_z_ + extra_rel;
  const double safe_z = std::max({
    last_lz + task1_transit_raise_above_lift_,
    goal_l0.point.z + task1_basket_approach_min_z_,
    last_gz + 0.14,
    release_z + 0.06});
  const double yaw_place = task1_arm_link8_yaw(used.yaw);

  const std::string gnd_id = "task1_ground_guard";
  add_ground_guard(gnd_id);

  // Rise
  auto rise_pose = make_topdown_pose(last_cx, last_cy, safe_z, yaw_place);
  if (!cartesian_move(rise_pose, 0.72)) {
    if (!move_arm_to_pose(rise_pose)) {
      RCLCPP_ERROR(node_->get_logger(), "Task1: rise to transit height failed");
      remove_collision_objects({gnd_id}); return;
    }
  }

  // Horizontal transit (multi-waypoint Cartesian for smoothness)
  auto hover_basket = make_topdown_pose(goal_l0.point.x, goal_l0.point.y, safe_z, yaw_place);
  std::vector<geometry_msgs::msg::Pose> to_basket;
  constexpr int kSteps = 5;
  for (int s = 1; s <= kSteps; ++s) {
    const double t = static_cast<double>(s) / kSteps;
    to_basket.push_back(make_topdown_pose(
      last_cx + t * (goal_l0.point.x - last_cx),
      last_cy + t * (goal_l0.point.y - last_cy),
      safe_z, yaw_place));
  }
  if (!cartesian_follow_waypoints(to_basket, 0.85,
      task1_transit_xy_vel_scale_, task1_transit_xy_acc_scale_, 0.003)) {
    RCLCPP_WARN(node_->get_logger(), "Task1: Cartesian transit failed — using OMPL");
    if (!move_arm_to_pose(hover_basket)) {
      RCLCPP_ERROR(node_->get_logger(), "Task1: move to basket failed");
      remove_collision_objects({gnd_id}); return;
    }
  }

  // --- Place ---
  remove_collision_objects({gnd_id});

  auto at_release = make_topdown_pose(goal_l0.point.x, goal_l0.point.y, release_z, yaw_place);
  if (!cartesian_follow_waypoints({at_release}, 0.72,
      task1_place_descend_vel_scale_, task1_place_descend_acc_scale_, 0.001)) {
    if (!cartesian_move(at_release, 0.72)) {
      RCLCPP_ERROR(node_->get_logger(), "Task1: descend to place failed"); return;
    }
  }

  open_gripper();

  if (!cartesian_follow_waypoints({hover_basket}, 0.55,
      task1_place_descend_vel_scale_, task1_place_descend_acc_scale_, 0.001)) {
    (void)move_arm_to_pose(hover_basket);
  }

  remove_collision_objects({coll_obj_id});
  RCLCPP_INFO(node_->get_logger(), "Task1 completed");
}

// ====================== Task 2 / Task 3 stubs =========================


void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  response->mystery_object_num = -1;

  if (request->ref_object_points.size() != 2U) {
    RCLCPP_ERROR(node_->get_logger(),
      "Task2: expected exactly 2 reference points, got %zu",
      request->ref_object_points.size());
    return;
  }

  geometry_msgs::msg::PointStamped ref1_l0, ref2_l0, mystery_l0;
  if (!transform_point_to_link0(request->ref_object_points[0], ref1_l0) ||
      !transform_point_to_link0(request->ref_object_points[1], ref2_l0) ||
      !transform_point_to_link0(request->mystery_object_point, mystery_l0))
  {
    RCLCPP_ERROR(node_->get_logger(), "Task2: TF to panda_link0 failed");
    return;
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task2 start: ref1=(%.3f,%.3f,%.3f) ref2=(%.3f,%.3f,%.3f) mystery=(%.3f,%.3f,%.3f)",
    ref1_l0.point.x, ref1_l0.point.y, ref1_l0.point.z,
    ref2_l0.point.x, ref2_l0.point.y, ref2_l0.point.z,
    mystery_l0.point.x, mystery_l0.point.y, mystery_l0.point.z);

  auto observe_and_classify =
    [this](const geometry_msgs::msg::PointStamped &obj_l0,
           const char *name,
           Task2ObservedObject &obs) -> bool
    {
      obs = Task2ObservedObject{};

      const double yaw_to_center = std::atan2(obj_l0.point.y, obj_l0.point.x);

      
      const double obs_z_high = std::max(0.55, obj_l0.point.z + 0.48);
      // 再到正式观察高度
      const double obs_z_low  = std::max(0.48, obj_l0.point.z + 0.40);

      const auto obs_pose_high = make_topdown_pose(obj_l0.point.x, obj_l0.point.y, obs_z_high, yaw_to_center);
      const auto obs_pose_low  = make_topdown_pose(obj_l0.point.x, obj_l0.point.y, obs_z_low,  yaw_to_center);

      if (!move_arm_to_pose(obs_pose_high)) {
        RCLCPP_WARN(node_->get_logger(),
          "Task2: move to high observation pose failed for %s", name);
        return false;
      }

      if (!move_arm_to_pose(obs_pose_low)) {
        RCLCPP_WARN(node_->get_logger(),
          "Task2: move to low observation pose failed for %s", name);
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1400));

      std::uint64_t seq_before = 0;
      { std::lock_guard<std::mutex> lk(cloud_mutex_); seq_before = g_cloud_sequence_; }

      const auto wait_ms = std::chrono::milliseconds(
        static_cast<int>(std::max(1.0, cloud_wait_timeout_sec_) * 1000.0));

      if (!wait_for_cloud(wait_ms, seq_before, nullptr)) {
        if (cloud_recently_valid(std::chrono::seconds(3))) {
          RCLCPP_WARN(node_->get_logger(),
            "Task2: no new cloud after moving to %s; using latest", name);
        } else {
          RCLCPP_ERROR(node_->get_logger(), "Task2: no fresh cloud for %s", name);
          return false;
        }
      }

      PointCPtr cloud_cam;
      std::string cloud_frame;
      std::uint64_t cloud_seq = 0;
      if (!get_cloud_snapshot(cloud_cam, cloud_frame, cloud_seq) || !cloud_cam || cloud_cam->empty()) {
        RCLCPP_ERROR(node_->get_logger(), "Task2: empty cloud for %s", name);
        return false;
      }

      PointC cloud_l0;
      if (!transform_cloud_to_frame(*cloud_cam, cloud_frame, "panda_link0", cloud_l0)) {
        RCLCPP_ERROR(node_->get_logger(), "Task2: cloud TF failed for %s", name);
        return false;
      }

      constexpr double kCoarseRadius = 0.22;
      extract_task2_roi_xy_only(cloud_l0, obj_l0.point, kCoarseRadius, obs.coarse_roi);

      if (obs.coarse_roi.size() < 30U) {
        RCLCPP_WARN(node_->get_logger(),
          "Task2: coarse ROI too small for %s (%zu)", name, obs.coarse_roi.size());
        return false;
      }

      if (!task2_find_refined_center_from_roi(
            obs.coarse_roi, obs.refined_center, obs.top_band, obs.surface_roi))
      {
        RCLCPP_WARN(node_->get_logger(),
          "Task2: failed to refine center for %s (coarse=%zu)",
          name, obs.coarse_roi.size());
        return false;
      }

      obs.cls = classify_task2_shape_from_surface(obs.surface_roi, obs.refined_center);
      obs.ok = (obs.cls.label != Task2ShapeLabel::UNKNOWN);

      RCLCPP_INFO(node_->get_logger(),
        "Task2 %s: coarse=%zu top=%zu surface=%zu refined=(%.3f,%.3f,%.3f) "
        "label=%s nought=%.5f cross=%.5f",
        name,
        obs.coarse_roi.size(),
        obs.top_band.size(),
        obs.surface_roi.size(),
        obs.refined_center.x, obs.refined_center.y, obs.refined_center.z,
        task2_label_str(obs.cls.label),
        obs.cls.nought_cost, obs.cls.cross_cost);

      return obs.ok;
    };

  Task2ObservedObject ref1_obs, ref2_obs, mystery_obs;

  const bool ok1 = observe_and_classify(ref1_l0, "ref1", ref1_obs);
  const bool ok2 = observe_and_classify(ref2_l0, "ref2", ref2_obs);
  const bool ok3 = observe_and_classify(mystery_l0, "mystery", mystery_obs);


    
  if (ok1 && ok2 && ref1_obs.cls.label == ref2_obs.cls.label) {
    RCLCPP_WARN(node_->get_logger(),
      "Task2: ref1/ref2 classified as same label (%s). Re-observing both references.",
      task2_label_str(ref1_obs.cls.label));

    Task2ObservedObject ref1_retry, ref2_retry;
    const bool ok1_retry = observe_and_classify(ref1_l0, "ref1_retry", ref1_retry);
    const bool ok2_retry = observe_and_classify(ref2_l0, "ref2_retry", ref2_retry);

    if (ok1_retry) ref1_obs = ref1_retry;
    if (ok2_retry) ref2_obs = ref2_retry;
  }









  if (!ok1 || !ok2 || !ok3) {
    RCLCPP_ERROR(node_->get_logger(),
      "Task2: failed to observe/classify all objects (ref1=%d ref2=%d mystery=%d)",
      static_cast<int>(ok1), static_cast<int>(ok2), static_cast<int>(ok3));
    return;
  }

  if (ref1_obs.cls.label != ref2_obs.cls.label) {
    if (mystery_obs.cls.label == ref1_obs.cls.label) {
      response->mystery_object_num = 1;
    } else if (mystery_obs.cls.label == ref2_obs.cls.label) {
      response->mystery_object_num = 2;
    }
  }

  if (response->mystery_object_num == -1) {
    double ref1_match_cost = std::numeric_limits<double>::infinity();
    double ref2_match_cost = std::numeric_limits<double>::infinity();

    if (ref1_obs.cls.label == Task2ShapeLabel::NOUGHT) ref1_match_cost = mystery_obs.cls.nought_cost;
    else if (ref1_obs.cls.label == Task2ShapeLabel::CROSS) ref1_match_cost = mystery_obs.cls.cross_cost;

    if (ref2_obs.cls.label == Task2ShapeLabel::NOUGHT) ref2_match_cost = mystery_obs.cls.nought_cost;
    else if (ref2_obs.cls.label == Task2ShapeLabel::CROSS) ref2_match_cost = mystery_obs.cls.cross_cost;

    if (std::isfinite(ref1_match_cost) && std::isfinite(ref2_match_cost)) {
      response->mystery_object_num = (ref1_match_cost <= ref2_match_cost) ? 1 : 2;
      RCLCPP_WARN(node_->get_logger(),
        "Task2: fallback by match cost ref1=%.5f ref2=%.5f -> answer=%ld",
        ref1_match_cost, ref2_match_cost, response->mystery_object_num);
    }
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task2 completed: ref1=%s ref2=%s mystery=%s -> answer=%ld",
    task2_label_str(ref1_obs.cls.label),
    task2_label_str(ref2_obs.cls.label),
    task2_label_str(mystery_obs.cls.label),
    response->mystery_object_num);
}










void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;

  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();

  RCLCPP_INFO(node_->get_logger(), "Task3 start");

  std::vector<Task3Object> objects;
  std::vector<Task3Obstacle> obstacles;

  bool basket_found = false;
  double best_basket_score = 1e9;

  geometry_msgs::msg::Point basket_center;
  basket_center.x = 0.35;
  basket_center.y = 0.35;
  basket_center.z = 0.05;

  auto is_task3_black = [](const PointT &p) -> bool {
    const int r = static_cast<int>(p.r);
    const int g = static_cast<int>(p.g);
    const int b = static_cast<int>(p.b);
    const int mx = std::max({r, g, b});
    const int mn = std::min({r, g, b});

    
    return mx < 95 && (mx - mn) < 35;
  };


  auto is_task3_brown = [](const PointT &p) -> bool {
    return p.r > 65 && p.r < 180 &&
           p.g > 20 && p.g < 115 &&
           p.b > 10 && p.b < 100 &&
           p.r > p.g * 1.25 &&
           p.g >= p.b * 0.65;
  };

  auto is_task3_grass = [](const PointT &p) -> bool {
    return p.g > p.r * 1.18 && p.g > p.b * 1.18 && p.g > 60;
  };

  auto is_task3_shape_color = [&](const PointT &p) -> bool {
    if (is_task3_black(p)) return false;
    if (is_task3_brown(p)) return false;
    if (is_task3_grass(p)) return false;

    const int mx = std::max({static_cast<int>(p.r), static_cast<int>(p.g), static_cast<int>(p.b)});
    const int mn = std::min({static_cast<int>(p.r), static_cast<int>(p.g), static_cast<int>(p.b)});

    
    
    if (mx < 95 && (mx - mn) < 45) return false;

    
    if ((mx - mn) < 25 && mx < 150) return false;
   
    return true;
  };

  auto classify_by_hole = [&](const PointC &cl) -> Task2ShapeLabel {
    if (cl.size() < 24U) return Task2ShapeLabel::UNKNOWN;

    double min_x, max_x, min_y, max_y, min_z, max_z;
    cloud_bounds(cl, min_x, max_x, min_y, max_y, min_z, max_z);

    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double dz = max_z - min_z;

    if (dx < 0.040 || dy < 0.040) return Task2ShapeLabel::UNKNOWN;
    if (dx > 0.240 || dy > 0.240) return Task2ShapeLabel::UNKNOWN;
    if (dz > 0.13) return Task2ShapeLabel::UNKNOWN;

    const double cx = 0.5 * (min_x + max_x);
    const double cy = 0.5 * (min_y + max_y);
    const double side = std::max(dx, dy);
    const double aspect = std::min(dx, dy) / std::max(dx, dy);

    if (aspect < 0.40) return Task2ShapeLabel::UNKNOWN;

    int inner = 0;
    int mid = 0;
    int outer = 0;

    for (const auto &p : cl.points) {
      const double ux = std::abs((p.x - cx) / (0.5 * dx));
      const double uy = std::abs((p.y - cy) / (0.5 * dy));
      const double m = std::max(ux, uy);

      if (ux < 0.28 && uy < 0.28) ++inner;
      if (m < 0.62) ++mid;
      if (m > 0.70) ++outer;
    }

    const double n = static_cast<double>(cl.size());
    const double inner_ratio = static_cast<double>(inner) / n;
    const double mid_ratio = static_cast<double>(mid) / n;
    const double outer_ratio = static_cast<double>(outer) / n;

    if (side > 0.045 && side < 0.230 &&
        inner_ratio < 0.085 &&
        outer_ratio > 0.22) {
      return Task2ShapeLabel::NOUGHT;
    }

    if (side > 0.060 && side < 0.230 &&
        inner_ratio > 0.070 &&
        mid_ratio > 0.16) {
      return Task2ShapeLabel::CROSS;
    }

    return Task2ShapeLabel::UNKNOWN;
  };

  auto too_close_to_basket = [&](double x, double y, double side) -> bool {
    const double db_found = std::hypot(x - basket_center.x, y - basket_center.y);
    const double db_default = std::hypot(x - 0.35, y - 0.35);

    if (basket_found && db_found < 0.26) return true;

    
    if (db_default < 0.24 && side > 0.045) return true;

    return false;
  };

  auto process_shape_cluster = [&](const PointC &cl_raw) {
    if (cl_raw.size() < 12U) return;

      int black_like = 0;
      int brown_like = 0;
      int grass_like = 0;

      for (const auto &p : cl_raw.points) {
        if (is_task3_black(p)) ++black_like;
        if (is_task3_brown(p)) ++brown_like;
        if (is_task3_grass(p)) ++grass_like;
      }

      const double n_color = static_cast<double>(cl_raw.size());
      const double black_ratio = static_cast<double>(black_like) / n_color;
      const double brown_ratio = static_cast<double>(brown_like) / n_color;
      const double grass_ratio = static_cast<double>(grass_like) / n_color;

      
      if (black_ratio > 0.18 || brown_ratio > 0.18 || grass_ratio > 0.35) {
        RCLCPP_INFO(node_->get_logger(),
          "Task3 reject non-shape cluster: size=%zu black=%.2f brown=%.2f grass=%.2f",
          cl_raw.size(), black_ratio, brown_ratio, grass_ratio);
        return;
      }

    PointC cl = cl_raw;

    double min_x, max_x, min_y, max_y, min_z, max_z;
    cloud_bounds(cl, min_x, max_x, min_y, max_y, min_z, max_z);

    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double dz = max_z - min_z;
    const double side = std::max(dx, dy);
    const double aspect = std::min(dx, dy) / std::max(dx, dy);

    if (dx < 0.035 || dy < 0.035) return;
    if (dx > 0.230 || dy > 0.230) return;
    if (dz < 0.002 || dz > 0.135) return;
    if (aspect < 0.36) return;

    std::vector<double> xs, ys, zs;
    xs.reserve(cl.size());
    ys.reserve(cl.size());
    zs.reserve(cl.size());

    for (const auto &p : cl.points) {
      xs.push_back(p.x);
      ys.push_back(p.y);
      zs.push_back(p.z);
    }

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(zs.begin(), zs.end());

    geometry_msgs::msg::Point c;
    c.x = percentile_from_sorted(xs, 0.50);
    c.y = percentile_from_sorted(ys, 0.50);
    c.z = percentile_from_sorted(zs, 0.50);

    const double r_base = std::hypot(c.x, c.y);
    if (r_base < 0.22 || r_base > 0.93) return;

    if (too_close_to_basket(c.x, c.y, side)) return;

    geometry_msgs::msg::Point refined_center;
    double estimated_cell = 0.040;
    Task2ShapeResult tpl_cls = classify_task3_cluster_multiscale(cl, &refined_center, &estimated_cell);
    Task2ShapeLabel geo_label = classify_by_hole(cl);

    Task2ShapeLabel final_label = Task2ShapeLabel::UNKNOWN;

    if (geo_label != Task2ShapeLabel::UNKNOWN) {
      final_label = geo_label;
    } else if (tpl_cls.label != Task2ShapeLabel::UNKNOWN) {
      const double best = std::min(tpl_cls.nought_cost, tpl_cls.cross_cost);
      const double worst = std::max(tpl_cls.nought_cost, tpl_cls.cross_cost);

      if (std::isfinite(best) && best < 0.045 && (worst - best) > 0.0015) {
        final_label = tpl_cls.label;
      }
    }

    if (final_label == Task2ShapeLabel::UNKNOWN) return;

    if (final_label == Task2ShapeLabel::NOUGHT) {
      if (side < 0.065 || aspect < 0.45) return;
    }

    if (final_label == Task2ShapeLabel::CROSS) {
      if (side < 0.055 || cl.size() < 24U) return;
    }

    Task3Object obj;
    obj.center = refined_center;
    obj.center.z = c.z;
    obj.points = cl;
    obj.label = final_label;
    obj.observations = 1;

    RCLCPP_INFO(node_->get_logger(),
      "Task3 candidate object: center=(%.3f, %.3f, %.3f) r=%.3f size=%zu bbox=(%.3f,%.3f,%.3f) aspect=%.2f label=%s geo=%s nought=%.5f cross=%.5f",
      obj.center.x, obj.center.y, obj.center.z,
      r_base,
      obj.points.size(),
      dx, dy, dz, aspect,
      task2_label_str(obj.label),
      task2_label_str(geo_label),
      tpl_cls.nought_cost,
      tpl_cls.cross_cost);

    merge_task3_object(objects, obj);
  };

  const std::vector<std::pair<double, double>> scan_xy = {
    { 0.55,  0.00},
    { 0.39,  0.39},
    { 0.00,  0.55},
    {-0.39,  0.39},
    {-0.55,  0.00},
    {-0.39, -0.39},
    { 0.00, -0.55},
    { 0.39, -0.39}
  };

  for (std::size_t si = 0; si < scan_xy.size(); ++si) {
    const double sx = scan_xy[si].first;
    const double sy = scan_xy[si].second;
    const double yaw = std::atan2(sy, sx);

    const auto scan_pose = make_topdown_pose(sx, sy, 0.74, yaw);

    if (!move_arm_to_pose(scan_pose)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: failed scan pose %zu", si);
      continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    std::uint64_t seq_before = 0;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      seq_before = g_cloud_sequence_;
    }

    const auto wait_ms = std::chrono::milliseconds(
      static_cast<int>(std::max(1.0, cloud_wait_timeout_sec_) * 1000.0));

    if (!wait_for_cloud(wait_ms, seq_before, nullptr)) {
      if (!cloud_recently_valid(std::chrono::seconds(3))) {
        RCLCPP_WARN(node_->get_logger(), "Task3: no fresh cloud at scan %zu", si);
        continue;
      }
    }

    PointCPtr cloud_cam;
    std::string cloud_frame;
    std::uint64_t cloud_seq = 0;

    if (!get_cloud_snapshot(cloud_cam, cloud_frame, cloud_seq) || !cloud_cam || cloud_cam->empty()) {
      RCLCPP_WARN(node_->get_logger(), "Task3: empty cloud at scan %zu", si);
      continue;
    }

    PointC cloud_l0;
    if (!transform_cloud_to_frame(*cloud_cam, cloud_frame, "panda_link0", cloud_l0)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: cloud TF failed at scan %zu", si);
      continue;
    }

    PointC shape_pts, obstacle_pts, basket_pts;
    shape_pts.header = cloud_l0.header;
    obstacle_pts.header = cloud_l0.header;
    basket_pts.header = cloud_l0.header;

    for (const auto &p : cloud_l0.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

      const double r_base = std::hypot(p.x, p.y);
      if (r_base < 0.21 || r_base > 0.96) continue;

      if (p.z < 0.018 || p.z > 0.18) continue;

      if (is_task3_black(p)) {
        obstacle_pts.push_back(p);
      } else if (is_task3_brown(p)) {
        basket_pts.push_back(p);
      } else if (is_task3_shape_color(p)) {
        if (std::hypot(p.x - basket_center.x, p.y - basket_center.y) > 0.20 &&
            std::hypot(p.x - 0.35, p.y - 0.35) > 0.20) {
          shape_pts.push_back(p);
        }
      }
    }

    if (basket_pts.size() > 1200U) {
      PointC basket_vox;
      voxel_filter_xy(basket_pts, basket_vox, 0.010);

      std::vector<PointC> basket_clusters;
      simple_cluster_xy(basket_vox, 0.055, basket_clusters);

      for (const auto &bc : basket_clusters) {
        if (bc.size() < 80U) continue;

        double min_x, max_x, min_y, max_y, min_z, max_z;
        cloud_bounds(bc, min_x, max_x, min_y, max_y, min_z, max_z);

        const double bx = max_x - min_x;
        const double by = max_y - min_y;
        const double bz = max_z - min_z;
        const double area = bx * by;
        const double aspect = std::min(bx, by) / std::max(bx, by);

        const double cx = 0.5 * (min_x + max_x);
        const double cy = 0.5 * (min_y + max_y);

        const bool size_ok =
          bx > 0.30 && bx < 0.42 &&
          by > 0.30 && by < 0.42;

        const bool square_ok = aspect > 0.78;

        const bool height_ok =
          min_z > -0.015 &&
          min_z < 0.040 &&
          max_z > 0.025 &&
          max_z < 0.110 &&
          bz > 0.018 &&
          bz < 0.095;

        const bool area_ok = area > 0.085 && area < 0.180;

        const bool not_robot_base =
          std::hypot(cx, cy) > 0.22 &&
          std::hypot(cx, cy) < 0.95;

        if (!(size_ok && square_ok && height_ok && area_ok && not_robot_base)) {
          RCLCPP_INFO(node_->get_logger(),
            "Task3 basket candidate rejected: points=%zu center=(%.3f,%.3f) size=(%.3f,%.3f,%.3f) aspect=%.2f z=(%.3f,%.3f)",
            bc.size(), cx, cy, bx, by, bz, aspect, min_z, max_z);
          continue;
        }

        const double score =
          2.0 * std::abs(bx - 0.35) +
          2.0 * std::abs(by - 0.35) +
          1.0 * std::abs(bz - 0.05) +
          0.5 * std::abs(aspect - 1.0) -
          0.000001 * static_cast<double>(bc.size());

        if (score < best_basket_score) {
          best_basket_score = score;
          basket_center.x = cx;
          basket_center.y = cy;
          basket_center.z = 0.05;
          basket_found = true;

          RCLCPP_INFO(node_->get_logger(),
            "Task3 basket ACCEPTED: points=%zu center=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) score=%.4f",
            bc.size(),
            basket_center.x, basket_center.y, basket_center.z,
            bx, by, bz, score);
        }
      }
    }

    PointC shape_vox, obstacle_vox;
    voxel_filter_xy(shape_pts, shape_vox, 0.005);
    voxel_filter_xy(obstacle_pts, obstacle_vox, 0.009);

    std::vector<PointC> shape_clusters;
    simple_cluster_xy(shape_vox, 0.040, shape_clusters);

    RCLCPP_INFO(node_->get_logger(),
      "Task3 scan %zu raw_shape=%zu vox_shape=%zu clusters=%zu",
      si, shape_pts.size(), shape_vox.size(), shape_clusters.size());

    for (const auto &cl : shape_clusters) {
      double min_x, max_x, min_y, max_y, min_z, max_z;
      cloud_bounds(cl, min_x, max_x, min_y, max_y, min_z, max_z);

      const double dx = max_x - min_x;
      const double dy = max_y - min_y;

      if ((dx > 0.24 || dy > 0.24) && cl.size() > 140U) {
        PointC sub_vox;
        voxel_filter_xy(cl, sub_vox, 0.005);

        std::vector<PointC> sub_clusters;
        simple_cluster_xy(sub_vox, 0.032, sub_clusters);

        for (const auto &sub : sub_clusters) {
          process_shape_cluster(sub);
        }
      } else {
        process_shape_cluster(cl);
      }
    }

    std::vector<PointC> obstacle_clusters;
    simple_cluster_xy(obstacle_vox, 0.050, obstacle_clusters);

    for (const auto &cl : obstacle_clusters) {
      if (cl.size() < 12U) continue;

      Task3Obstacle obs;
      cloud_bounds(cl, obs.min_x, obs.max_x, obs.min_y, obs.max_y, obs.min_z, obs.max_z);
      obs.observations = 1;

      if ((obs.max_x - obs.min_x) < 0.025 || (obs.max_y - obs.min_y) < 0.025) continue;
      if ((obs.max_x - obs.min_x) > 0.35 || (obs.max_y - obs.min_y) > 0.35) continue;

      merge_task3_obstacle(obstacles, obs);
    }

    RCLCPP_INFO(node_->get_logger(),
      "Task3 scan %zu: objects_so_far=%zu obstacles_so_far=%zu basket=%d",
      si, objects.size(), obstacles.size(), static_cast<int>(basket_found));
  }

  std::vector<Task3Object> merged_objects;
  for (const auto &obj : objects) {
    bool merged = false;

    for (auto &dst : merged_objects) {
      const double d = std::hypot(dst.center.x - obj.center.x, dst.center.y - obj.center.y);
      if (d < 0.115) {
        dst.points += obj.points;
        dst.center = centroid_of_cloud(dst.points);

        geometry_msgs::msg::Point refined_center;
        double estimated_cell = 0.040;

        Task2ShapeResult cls = classify_task3_cluster_multiscale(dst.points, &refined_center, &estimated_cell);
        Task2ShapeLabel geo = classify_by_hole(dst.points);

        if (geo != Task2ShapeLabel::UNKNOWN) dst.label = geo;
        else if (cls.label != Task2ShapeLabel::UNKNOWN) dst.label = cls.label;

        dst.observations += obj.observations;
        merged = true;
        break;
      }
    }

    if (!merged) merged_objects.push_back(obj);
  }

  objects.clear();

  for (const auto &obj : merged_objects) {
    double min_x, max_x, min_y, max_y, min_z, max_z;
    cloud_bounds(obj.points, min_x, max_x, min_y, max_y, min_z, max_z);

    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double side = std::max(dx, dy);

    if (obj.label == Task2ShapeLabel::UNKNOWN) continue;

    int black_like = 0;
    int brown_like = 0;
    int grass_like = 0;

    for (const auto &p : obj.points.points) {
      if (is_task3_black(p)) ++black_like;
      if (is_task3_brown(p)) ++brown_like;
      if (is_task3_grass(p)) ++grass_like;
    }

    const double nn = static_cast<double>(std::max<std::size_t>(1, obj.points.size()));
    const double black_ratio = static_cast<double>(black_like) / nn;
    const double brown_ratio = static_cast<double>(brown_like) / nn;
    const double grass_ratio = static_cast<double>(grass_like) / nn;

    if (black_ratio > 0.08 || brown_ratio > 0.10 || grass_ratio > 0.25) {
      RCLCPP_INFO(node_->get_logger(),
        "Task3 final reject non-shape: center=(%.3f,%.3f) black=%.2f brown=%.2f grass=%.2f",
        obj.center.x, obj.center.y, black_ratio, brown_ratio, grass_ratio);
      continue;
    }





    if (side < 0.040 || side > 0.250) continue;
    if (too_close_to_basket(obj.center.x, obj.center.y, side)) continue;

    objects.push_back(obj);
  }

  int n_nought = 0;
  int n_cross = 0;

  for (const auto &obj : objects) {
    if (obj.label == Task2ShapeLabel::NOUGHT) ++n_nought;
    else if (obj.label == Task2ShapeLabel::CROSS) ++n_cross;
  }

  const int total = n_nought + n_cross;
  const int most_common_count = std::max(n_nought, n_cross);

  response->total_num_shapes = total;
  response->num_most_common_shape = most_common_count;

  RCLCPP_INFO(node_->get_logger(),
    "Task3 detected FINAL: total=%d nought=%d cross=%d most_common=%d basket=(%.3f,%.3f,%.3f)",
    total, n_nought, n_cross, most_common_count,
    basket_center.x, basket_center.y, basket_center.z);

  if (total <= 0 || objects.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task3: no valid shapes detected");
    return;
  }

  Task2ShapeLabel target_label =
    (n_nought >= n_cross) ? Task2ShapeLabel::NOUGHT : Task2ShapeLabel::CROSS;

  std::vector<int> target_indices;

  for (std::size_t i = 0; i < objects.size(); ++i) {
    if (objects[i].label != target_label) continue;

    const double r_base = std::hypot(objects[i].center.x, objects[i].center.y);
    if (r_base < 0.23 || r_base > 0.91) continue;

    double min_x, max_x, min_y, max_y, min_z, max_z;
    cloud_bounds(objects[i].points, min_x, max_x, min_y, max_y, min_z, max_z);

    const double side = std::max(max_x - min_x, max_y - min_y);
    if (too_close_to_basket(objects[i].center.x, objects[i].center.y, side)) continue;

    bool inside_obstacle = false;
    double min_obstacle_dist = 999.0;

    for (const auto &o : obstacles) {
      const double ox = 0.5 * (o.min_x + o.max_x);
      const double oy = 0.5 * (o.min_y + o.max_y);

      min_obstacle_dist = std::min(
        min_obstacle_dist,
        std::hypot(objects[i].center.x - ox, objects[i].center.y - oy));

      const double pad = 0.025;

      const bool inside_xy =
        objects[i].center.x > o.min_x - pad &&
        objects[i].center.x < o.max_x + pad &&
        objects[i].center.y > o.min_y - pad &&
        objects[i].center.y < o.max_y + pad;

      if (inside_xy) {
        inside_obstacle = true;
        break;
      }
    }

    if (inside_obstacle) continue;

    RCLCPP_INFO(node_->get_logger(),
      "Task3 target candidate: idx=%zu center=(%.3f,%.3f) points=%zu obstacle_dist=%.3f",
      i,
      objects[i].center.x,
      objects[i].center.y,
      objects[i].points.size(),
      min_obstacle_dist);

    target_indices.push_back(static_cast<int>(i));
  }

  std::sort(target_indices.begin(), target_indices.end(),
    [&](int a, int b) {
      const auto &A = objects[static_cast<std::size_t>(a)];
      const auto &B = objects[static_cast<std::size_t>(b)];

      double min_da = 999.0;
      double min_db = 999.0;

      for (const auto &o : obstacles) {
        const double ox = 0.5 * (o.min_x + o.max_x);
        const double oy = 0.5 * (o.min_y + o.max_y);

        min_da = std::min(min_da, std::hypot(A.center.x - ox, A.center.y - oy));
        min_db = std::min(min_db, std::hypot(B.center.x - ox, B.center.y - oy));
      }

      if (std::abs(min_da - min_db) > 0.03) {
        return min_da > min_db;  
      }

      return A.points.size() > B.points.size();  
    });

  if (target_indices.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task3: no target candidates selected");
    return;
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task3: selected %zu possible targets. First idx=%d",
    target_indices.size(), target_indices.front());

  Task3Object target = objects[static_cast<std::size_t>(target_indices.front())];
  const std::string target_shape =
    target.label == Task2ShapeLabel::NOUGHT ? "nought" : "cross";

  double tgt_min_x, tgt_max_x, tgt_min_y, tgt_max_y, tgt_min_z, tgt_max_z;
  cloud_bounds(target.points, tgt_min_x, tgt_max_x, tgt_min_y, tgt_max_y, tgt_min_z, tgt_max_z);

  const double target_ground_z = std::max(0.0, tgt_min_z - 0.010);
  const double target_top_z = tgt_max_z;

  geometry_msgs::msg::Point grasp_center = target.center;
  grasp_center.z = target_ground_z;

  RCLCPP_INFO(node_->get_logger(),
    "Task3 target: %s at center=(%.3f, %.3f, %.3f) ground_z=%.3f top_z=%.3f size=%zu",
    target_shape.c_str(),
    target.center.x, target.center.y, target.center.z,
    target_ground_z, target_top_z,
    target.points.size());

  std::vector<std::string> collision_ids;

  for (std::size_t i = 0; i < obstacles.size(); ++i) {
    const auto &o = obstacles[i];

    const double pad = 0.040;
    const double sx = std::max(0.04, o.max_x - o.min_x + 2.0 * pad);
    const double sy = std::max(0.04, o.max_y - o.min_y + 2.0 * pad);
    const double sz = std::max(0.05, o.max_z - o.min_z + 0.040);

    const double cx = 0.5 * (o.min_x + o.max_x);
    const double cy = 0.5 * (o.min_y + o.max_y);
    const double cz = 0.5 * (o.min_z + o.max_z) + 0.015;

    //if (std::hypot(cx - target.center.x, cy - target.center.y) < 0.13) continue;

    std::ostringstream oss;
    oss << "task3_obstacle_" << i;
    const std::string id = oss.str();

    add_collision_box(id, cx, cy, cz, sx, sy, sz);
    collision_ids.push_back(id);
  }

  add_ground_guard("task3_ground_guard");
  collision_ids.push_back("task3_ground_guard");

  std::vector<Task1GraspCandidate> candidates;
  generate_task1_candidates(target.points, grasp_center, target_shape, candidates);

  if (target.label == Task2ShapeLabel::NOUGHT) {

    double bx_min, bx_max, by_min, by_max, bz_min, bz_max;
    cloud_bounds(target.points, bx_min, bx_max, by_min, by_max, bz_min, bz_max);

    const double side = std::max(bx_max - bx_min, by_max - by_min);
    const double estimated_cell = std::clamp(side / 5.0, 0.020, 0.040);

    const double offset = 2.0 * estimated_cell;
    const double close_w = std::clamp(
      estimated_cell - task1_gripper_closing_margin_,
      gripper_min_width_,
      gripper_max_width_);
    

    Task1GraspCandidate a;
    a.x = target.center.x + offset;
    a.y = target.center.y;
    a.yaw = 0.0;
    a.close_width = close_w;
    candidates.insert(candidates.begin(), a);

    Task1GraspCandidate b = a;
    b.x = target.center.x - offset;
    b.yaw = kPi;
    candidates.insert(candidates.begin(), b);

    Task1GraspCandidate c = a;
    c.x = target.center.x;
    c.y = target.center.y + offset;
    c.yaw = 0.5 * kPi;
    candidates.insert(candidates.begin(), c);

    Task1GraspCandidate d = c;
    d.y = target.center.y - offset;
    d.yaw = -0.5 * kPi;
    candidates.insert(candidates.begin(), d);
  }

  if (candidates.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task3: no grasp candidates");
    remove_collision_objects(collision_ids);
    return;
  }

  bool grasp_ok = false;
  Task1GraspCandidate used;
  double last_x = 0.0;
  double last_y = 0.0;
  double last_z = 0.0;

  constexpr double kObjHeight = 0.040;

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    auto c = candidates[i];

    const double dc = std::hypot(c.x - target.center.x, c.y - target.center.y);

    double bb_min_x, bb_max_x, bb_min_y, bb_max_y, bb_min_z, bb_max_z;
    cloud_bounds(target.points, bb_min_x, bb_max_x, bb_min_y, bb_max_y, bb_min_z, bb_max_z);

    const double target_side = std::max(bb_max_x - bb_min_x, bb_max_y - bb_min_y);
    const double max_dc = std::max(0.050, 0.70 * target_side);

    if (dc > max_dc) continue;

    const double yaw8 = task1_arm_link8_yaw(c.yaw);

    if (!open_gripper()) continue;

    const double grasp_z_nominal = target_ground_z + 0.120;
    const double grasp_z_final = grasp_z_nominal + std::max(0.0, task1_grasp_backoff_z_);

    const double pre_floor = target_top_z + 0.14;
    const double pre_z = std::max({
      target_ground_z + task1_pregrasp_offset_z_,
      grasp_z_nominal + std::max(0.06, task1_approach_last_delta_z_ + 0.05),
      pre_floor
    });

    const double hover_z = std::min(0.54, std::max(pre_z + 0.06, pre_floor + 0.04));
    const double mid_z = std::min(hover_z - 0.012,
                                  std::max(pre_z + 0.028, grasp_z_nominal + 0.095));

    if (target.label == Task2ShapeLabel::CROSS) {
      c.close_width = std::clamp(c.close_width, 0.020, 0.040);
    } else {
      c.close_width = std::clamp(c.close_width, 0.020, 0.045);
    }

    RCLCPP_INFO(node_->get_logger(),
      "Task3 grasp try %zu: xy=(%.3f,%.3f) dc=%.3f yaw=%.1f close=%.4f hover=%.3f pre=%.3f grasp=%.3f",
      i, c.x, c.y, dc, c.yaw * 180.0 / kPi,
      c.close_width, hover_z, pre_z, grasp_z_final);

    if (!move_arm_to_pose(make_topdown_pose(c.x, c.y, hover_z, yaw8))) {
      RCLCPP_WARN(node_->get_logger(), "Task3: hover failed candidate %zu", i);
      continue;
    }

    std::vector<geometry_msgs::msg::Pose> drop;
    if (mid_z > pre_z + 0.012 && mid_z + 0.01 < hover_z) {
      drop.push_back(make_topdown_pose(c.x, c.y, mid_z, yaw8));
    }
    drop.push_back(make_topdown_pose(c.x, c.y, pre_z, yaw8));

    if (!cartesian_follow_waypoints(drop, 0.78, 0.10, 0.15, 0.001)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: drop to pregrasp failed candidate %zu", i);
      continue;
    }

    std::vector<geometry_msgs::msg::Pose> desc_hi{
      make_topdown_pose(c.x, c.y, grasp_z_nominal + task1_approach_last_delta_z_, yaw8),
      make_topdown_pose(c.x, c.y, grasp_z_nominal + 0.012, yaw8)
    };

    if (!cartesian_follow_waypoints(desc_hi, 0.78,
        task1_descend_velocity_scale_, task1_descend_acceleration_scale_, 0.001)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: descend upper failed candidate %zu", i);
      continue;
    }

    std::vector<geometry_msgs::msg::Pose> desc_lo{
      make_topdown_pose(c.x, c.y, grasp_z_nominal + 0.012, yaw8),
      make_topdown_pose(c.x, c.y, grasp_z_final, yaw8)
    };

    if (!cartesian_follow_waypoints(desc_lo, 0.70, 0.035, 0.055, 0.0005)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: descend final failed candidate %zu", i);
      continue;
    }

    if (!close_gripper_for_grasp(c.close_width)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: close failed candidate %zu", i);
      continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(180, task1_grasp_settle_ms_)));

    const double lift_z1 = std::max(target_ground_z + 0.18, grasp_z_final + 0.055);

    if (!cartesian_move(make_topdown_pose(c.x, c.y, lift_z1, yaw8), 0.60)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: first lift failed candidate %zu", i);
      continue;
    }

    const double lift_z2 = std::max(target_ground_z + task1_lift_offset_z_, lift_z1 + 0.08);

    if (!cartesian_move(make_topdown_pose(c.x, c.y, lift_z2, yaw8), 0.60)) {
      RCLCPP_WARN(node_->get_logger(), "Task3: second lift failed candidate %zu", i);
      continue;
    }

    grasp_ok = true;
    used = c;
    last_x = c.x;
    last_y = c.y;
    last_z = lift_z2;
    break;
  }

  if (!grasp_ok) {
    RCLCPP_ERROR(node_->get_logger(), "Task3: failed to grasp target");
    remove_collision_objects(collision_ids);
    return;
  }

 

  // ===== Place into basket =====
  const double place_yaw = task1_arm_link8_yaw(used.yaw);

  
  const double safe_z = std::max({
    last_z + 0.25,
    basket_center.z + 0.62,
    0.72
  });

  const double release_z = basket_center.z + 0.23;

  auto high_above_object = make_topdown_pose(last_x, last_y, safe_z, place_yaw);
  auto high_above_basket = make_topdown_pose(basket_center.x, basket_center.y, safe_z, place_yaw);
  auto release_pose = make_topdown_pose(basket_center.x, basket_center.y, release_z, place_yaw);

  
  if (!move_arm_to_pose(high_above_object)) {
    RCLCPP_WARN(node_->get_logger(), "Task3: OMPL rise failed, trying Cartesian rise");

    if (!cartesian_move(high_above_object, 0.45)) {
      RCLCPP_ERROR(node_->get_logger(), "Task3: rise to safe z failed");
      remove_collision_objects(collision_ids);
      return;
    }
  }

  if (!move_arm_to_pose(high_above_basket)) {
    RCLCPP_ERROR(node_->get_logger(), "Task3: move above basket failed with obstacles enabled");
    remove_collision_objects(collision_ids);
    return;
  }

  
  remove_collision_objects(collision_ids);
  collision_ids.clear();

  bool place_ok = cartesian_follow_waypoints(
    {release_pose},
    0.45,
    task1_place_descend_vel_scale_,
    task1_place_descend_acc_scale_,
    0.001);

  if (!place_ok) {
    RCLCPP_WARN(node_->get_logger(),
      "Task3: descend to basket failed, opening gripper above basket anyway");
  }

 
  open_gripper();

  std::this_thread::sleep_for(std::chrono::milliseconds(400));

  (void)move_arm_to_pose(high_above_basket);
    
  
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  (void)cartesian_move(make_topdown_pose(basket_center.x, basket_center.y, safe_z, place_yaw), 0.40);

  remove_collision_objects(collision_ids);

  RCLCPP_INFO(node_->get_logger(),
    "Task3 completed: total_num_shapes=%ld num_most_common_shape=%ld",
    response->total_num_shapes,
    response->num_most_common_shape);
}
