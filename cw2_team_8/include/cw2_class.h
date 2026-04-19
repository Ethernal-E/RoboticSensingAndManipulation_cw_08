/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#ifndef CW2_CLASS_H_
#define CW2_CLASS_H_

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "cw2_world_spawner/srv/task1_service.hpp"
#include "cw2_world_spawner/srv/task2_service.hpp"
#include "cw2_world_spawner/srv/task3_service.hpp"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointC;
typedef PointC::Ptr PointCPtr;

struct Task1GraspCandidate
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double close_width = 0.0;
  double score = 0.0;
};

class cw2
{
public:
  explicit cw2(const rclcpp::Node::SharedPtr &node);

  // ---- Service callbacks ----
  void t1_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response);
  void t2_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response);
  void t3_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response);

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  // ---- ROS handles ----
  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<cw2_world_spawner::srv::Task1Service>::SharedPtr t1_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task2Service>::SharedPtr t2_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task3Service>::SharedPtr t3_service_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr color_cloud_sub_;
  rclcpp::CallbackGroup::SharedPtr pointcloud_callback_group_;
  rclcpp::CallbackGroup::SharedPtr task_service_callback_group_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> hand_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  mutable std::mutex cloud_mutex_;
  PointCPtr g_cloud_ptr;
  std::uint64_t g_cloud_sequence_ = 0;
  std::string g_input_pc_frame_id_;
  std::chrono::steady_clock::time_point g_cloud_receive_steady_ =
    std::chrono::steady_clock::time_point::min();

  std::string pointcloud_topic_;
  bool pointcloud_qos_reliable_ = false;

  // ======================= General tunables =======================
  double cloud_max_age_sec_ = 0.35;
  double cloud_wait_timeout_sec_ = 25.0;
  int spawn_settle_ms_ = 800;
  double gripper_open_width_ = 0.08;
  double gripper_max_width_ = 0.08;
  double gripper_min_width_ = 0.0;
  double blocked_close_width_tolerance_ = 0.040;

  // ====================== Task 1 tunables =========================
  double task1_roi_radius_xy_ = 0.14;
  double task1_roi_half_height_ = 0.05;
  double task1_surface_percentile_ = 0.72;
  double task1_ground_reject_height_ = 0.02;
  double task1_gripper_closing_margin_ = 0.018;
  double task1_candidate_strip_half_depth_ = 0.045;
  double task1_contact_side_threshold_ = 0.004;
  int task1_min_contact_points_per_side_ = 5;
  int task1_max_candidates_to_try_ = 6;
  int task1_n_yaw_samples_ = 20;
  double task1_pregrasp_offset_z_ = 0.18;
  double task1_grasp_offset_z_ = 0.125;
  double task1_lift_offset_z_ = 0.20;
  double task1_descend_velocity_scale_ = 0.085;
  double task1_descend_acceleration_scale_ = 0.14;
  double task1_approach_last_delta_z_ = 0.022;
  int task1_grasp_settle_ms_ = 180;
  double task1_grasp_backoff_z_ = 0.012;
  double task1_place_offset_z_ = 0.14;
  double task1_release_min_extra_z_ = 0.03;
  double task1_release_max_extra_z_ = 0.08;
  double task1_release_object_height_scale_ = 0.50;
  double task1_basket_approach_min_z_ = 0.30;
  double task1_transit_raise_above_lift_ = 0.12;
  double task1_transit_xy_vel_scale_ = 0.12;
  double task1_transit_xy_acc_scale_ = 0.18;
  double task1_place_descend_vel_scale_ = 0.06;
  double task1_place_descend_acc_scale_ = 0.10;
  double task1_ee_yaw_offset_rad_ = -2.356194490192345;

private:
  // ================= Common utilities (reusable) ==================
  void setup_arm_hand(
    moveit::planning_interface::MoveGroupInterface &arm,
    moveit::planning_interface::MoveGroupInterface &hand) const;

  // -- Transform --
  bool transform_point_to_link0(
    const geometry_msgs::msg::PointStamped &in,
    geometry_msgs::msg::PointStamped &out) const;

  bool transform_cloud_to_frame(
    const PointC &cloud_in,
    const std::string &input_frame,
    const std::string &target_frame,
    PointC &cloud_out) const;

  // -- Point cloud --
  bool wait_for_cloud(
    std::chrono::milliseconds timeout,
    std::uint64_t min_sequence_exclusive,
    std::uint64_t *latest_sequence) const;

  bool cloud_recently_valid(std::chrono::seconds max_age) const;

  bool get_cloud_snapshot(PointCPtr &cloud, std::string &frame_id, std::uint64_t &sequence) const;

  /** Extract points within a cylindrical ROI around @p center. */
  void extract_roi_points(
    const PointC &cloud_in_link0,
    const geometry_msgs::msg::Point &center,
    double radius_xy,
    double half_height_z,
    double min_z,
    PointC &roi_out) const;

  // -- Arm movement --
  bool execute_plan(
    moveit::planning_interface::MoveGroupInterface &group,
    const moveit::planning_interface::MoveGroupInterface::Plan &plan) const;

  bool move_arm_to_pose(const geometry_msgs::msg::Pose &target_pose) const;

  bool cartesian_move(const geometry_msgs::msg::Pose &target_pose, double min_fraction) const;

  bool cartesian_follow_waypoints(
    const std::vector<geometry_msgs::msg::Pose> &waypoints,
    double min_fraction,
    double vel_scale,
    double acc_scale,
    double eef_step) const;

  // -- Gripper --
  bool set_gripper(double width_m) const;
  double current_gripper_width() const;
  bool open_gripper() const;
  bool close_gripper_for_grasp(double target_width_m) const;

  // -- Collision scene helpers --
  void add_collision_box(
    const std::string &id,
    double cx, double cy, double cz,
    double sx, double sy, double sz);

  void add_ground_guard(const std::string &id);

  void remove_collision_objects(const std::vector<std::string> &ids);

  // -- Pose --
  geometry_msgs::msg::Pose make_topdown_pose(double x, double y, double z, double yaw_rad) const;

  // ================= Task 1 specifics =============================
  void generate_task1_candidates(
    const PointC &roi,
    const geometry_msgs::msg::Point &object_point_link0,
    const std::string &shape_type,
    std::vector<Task1GraspCandidate> &candidates_out) const;

  double task1_arm_link8_yaw(double grasp_model_yaw_rad) const;
};

#endif  // CW2_CLASS_H_
