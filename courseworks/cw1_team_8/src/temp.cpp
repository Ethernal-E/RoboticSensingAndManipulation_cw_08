/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire 
solution is contained within the cw1_team_<your_team_number> package */

#include "../include/cw1_class.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Quaternion.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

///////////////////////////////////////////////////////////////////////////////

cw1::cw1(const rclcpp::Node::SharedPtr &node): tf_buffer_(node->get_clock()),
  tf_listener_(tf_buffer_)
{
  /* class constructor */

  node_ = node;
  service_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  sensor_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // advertise solutions for coursework tasks
  t1_service_ = node_->create_service<cw1_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw1::t1_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t2_service_ = node_->create_service<cw1_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw1::t2_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t3_service_ = node_->create_service<cw1_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw1::t3_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);

  // Service and sensor callbacks use separate callback groups to align with the
  // current runtime architecture used in cw1_team_0.
  rclcpp::SubscriptionOptions joint_state_sub_options;
  joint_state_sub_options.callback_group = sensor_cb_group_;
  auto joint_state_qos = rclcpp::QoS(rclcpp::KeepLast(50));
  joint_state_qos.reliable();
  joint_state_qos.durability_volatile();
  joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", joint_state_qos,
    [this](const sensor_msgs::msg::JointState::ConstSharedPtr msg) {
      const int64_t stamp_ns =
        static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      latest_joint_state_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      joint_state_msg_count_.fetch_add(1, std::memory_order_relaxed);
    },
    joint_state_sub_options);

  rclcpp::SubscriptionOptions cloud_sub_options;
  cloud_sub_options.callback_group = sensor_cb_group_;
  auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(10));
  cloud_qos.reliable();
  cloud_qos.durability_volatile();
  cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/r200/camera/depth_registered/points", cloud_qos,
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {

      latest_cloud_ = msg;


      const int64_t stamp_ns =
        static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      latest_cloud_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      cloud_msg_count_.fetch_add(1, std::memory_order_relaxed);
    },
    cloud_sub_options);

  // Parameter declarations intentionally mirror cw1_team_0 for compatibility.
  const bool use_gazebo_gui = node_->declare_parameter<bool>("use_gazebo_gui", true);
  (void)use_gazebo_gui;
  enable_cloud_viewer_ = node_->declare_parameter<bool>("enable_cloud_viewer", false);
  move_home_on_start_ = node_->declare_parameter<bool>("move_home_on_start", false);
  use_path_constraints_ = node_->declare_parameter<bool>("use_path_constraints", false);
  use_cartesian_reach_ = node_->declare_parameter<bool>("use_cartesian_reach", false);
  allow_position_only_fallback_ = node_->declare_parameter<bool>(
    "allow_position_only_fallback", allow_position_only_fallback_);
  cartesian_eef_step_ = node_->declare_parameter<double>(
    "cartesian_eef_step", cartesian_eef_step_);
  cartesian_jump_threshold_ = node_->declare_parameter<double>(
    "cartesian_jump_threshold", cartesian_jump_threshold_);
  cartesian_min_fraction_ = node_->declare_parameter<double>(
    "cartesian_min_fraction", cartesian_min_fraction_);
  publish_programmatic_debug_ = node_->declare_parameter<bool>(
    "publish_programmatic_debug", publish_programmatic_debug_);
  enable_task1_snap_ = node_->declare_parameter<bool>("enable_task1_snap", false);
  return_home_between_pick_place_ = node_->declare_parameter<bool>(
    "return_home_between_pick_place", return_home_between_pick_place_);
  return_home_after_pick_place_ = node_->declare_parameter<bool>(
    "return_home_after_pick_place", return_home_after_pick_place_);
  pick_offset_z_ = node_->declare_parameter<double>("pick_offset_z", pick_offset_z_);
  task3_pick_offset_z_ = node_->declare_parameter<double>(
    "task3_pick_offset_z", task3_pick_offset_z_);
  task2_capture_enabled_ = node_->declare_parameter<bool>(
    "task2_capture_enabled", task2_capture_enabled_);
  task2_capture_dir_ = node_->declare_parameter<std::string>(
    "task2_capture_dir", task2_capture_dir_);
  place_offset_z_ = node_->declare_parameter<double>("place_offset_z", place_offset_z_);
  grasp_approach_offset_z_ = node_->declare_parameter<double>(
    "grasp_approach_offset_z", grasp_approach_offset_z_);
  post_grasp_lift_z_ = node_->declare_parameter<double>(
    "post_grasp_lift_z", post_grasp_lift_z_);
  gripper_grasp_width_ = node_->declare_parameter<double>(
    "gripper_grasp_width", gripper_grasp_width_);
  joint_state_wait_timeout_sec_ = node_->declare_parameter<double>(
    "joint_state_wait_timeout_sec", joint_state_wait_timeout_sec_);

  if (task2_capture_enabled_) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Template capture mode enabled, output dir: %s",
      task2_capture_dir_.c_str());
  }

  RCLCPP_INFO(node_->get_logger(), "cw1 template class initialised with compatibility scaffold");
}

///////////////////////////////////////////////////////////////////////////////
// Shared helper functions (used by Task 1 and Task 3)
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Execute the target already configured on a MoveIt planning group.
 * @param group Type: `moveit::planning_interface::MoveGroupInterface &`.
 * Meaning: the MoveIt group to execute, e.g. `panda_arm` or `hand`. The caller
 * should already have set a pose target or joint target on this group.
 * @return `true` if planning and execution both succeed.
 */
bool
cw1::execute_plan(
  moveit::planning_interface::MoveGroupInterface &group)
{
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  if (!static_cast<bool>(group.plan(plan))) {
    RCLCPP_ERROR(node_->get_logger(), "Planning failed");
    return false;
  }
  if (group.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node_->get_logger(), "Execution failed");
    return false;
  }
  return true;
}

/**
 * @brief Move the arm to a target end-effector pose.
 * @param arm Type: `moveit::planning_interface::MoveGroupInterface &`.
 * Meaning: the Panda arm MoveIt group used for planning and execution.
 * @param target Type: `const geometry_msgs::msg::Pose &`.
 * Meaning: desired end-effector pose; position is in metres and orientation is
 * a quaternion.
 * @param start_state Type: `const moveit::core::RobotState *`.
 * Meaning: optional planning start state; use `nullptr` to start from the
 * robot's current state.
 * @param target_link Type: `const char *`.
 * Meaning: name of the end-effector link that should reach `target`, usually
 * `"panda_hand"`.
 * @return `true` if the motion is planned and executed successfully.
 */
bool
cw1::move_arm_to_pose(
  moveit::planning_interface::MoveGroupInterface &arm,
  const geometry_msgs::msg::Pose &target,
  const moveit::core::RobotState *start_state,
  const char *target_link)
{
  // Use the live robot state by default, but allow callers to override it
  // when they need to plan from an explicit start configuration.
  if (start_state == nullptr)
    arm.setStartStateToCurrentState();
  else
    arm.setStartState(*start_state);

  arm.setPoseTarget(target, target_link);
  return execute_plan(arm);
}

/**
 * @brief Execute a short Cartesian move from the current pose to `target`.
 * @param arm Type: `moveit::planning_interface::MoveGroupInterface &`.
 * Meaning: the Panda arm MoveIt group used to compute the Cartesian path.
 * @param target Type: `const geometry_msgs::msg::Pose &`.
 * Meaning: final end-effector pose; position is in metres and orientation is a
 * quaternion.
 * @return `true` if a sufficiently complete Cartesian path is executed.
 */
bool
cw1::cartesian_move(
  moveit::planning_interface::MoveGroupInterface &arm,
  const geometry_msgs::msg::Pose &target)
{
  std::vector<geometry_msgs::msg::Pose> waypoints = {target};
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm.computeCartesianPath(
    waypoints, cartesian_eef_step_, cartesian_jump_threshold_, trajectory, true);
  if (fraction < cartesian_min_fraction_) {
    RCLCPP_ERROR(node_->get_logger(), "Cartesian fraction too low: %.3f", fraction);
    return false;
  }
  moveit::planning_interface::MoveGroupInterface::Plan cart_plan;
  cart_plan.trajectory_ = trajectory;
  auto result = arm.execute(cart_plan);
  arm.stop();
  return result == moveit::core::MoveItErrorCode::SUCCESS;
}

/**
 * @brief Open or close the gripper to a requested total width.
 * @param hand Type: `moveit::planning_interface::MoveGroupInterface &`.
 * Meaning: the hand MoveIt group controlling the two finger joints.
 * @param total_width Type: `double`.
 * Meaning: requested total distance between the two fingertips, measured in
 * metres; the function splits it equally across both fingers.
 * @return `true` if the finger joint motion is planned and executed.
 */
bool
cw1::set_gripper(
  moveit::planning_interface::MoveGroupInterface &hand,
  double total_width)
{
  const double per_finger = total_width / 2.0;
  std::vector<double> joint_values = hand.getCurrentJointValues();
  joint_values[0] = per_finger;
  joint_values[1] = per_finger;
  hand.setStartStateToCurrentState();
  hand.setJointValueTarget(joint_values);
  return execute_plan(hand);
}

/**
 * @brief Create a downward-facing grasp pose at `(x, y, z)`.
 * @param x Type: `double`. Meaning: target X coordinate in metres.
 * @param y Type: `double`. Meaning: target Y coordinate in metres.
 * @param z Type: `double`. Meaning: target Z coordinate in metres.
 * @return Pose whose palm faces downward for top-down approach motions.
 */
geometry_msgs::msg::Pose
cw1::make_pose(double x, double y, double z)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  tf2::Quaternion q;
  q.setRPY(M_PI, 0.0, 0.0);  // Rotate 180 degrees around X so the palm faces downward.
  q.normalize();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

///////////////////////////////////////////////////////////////////////////////

void
cw1::t1_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;
  RCLCPP_INFO(node_->get_logger(), "Task 1 callback triggered: starting pick-and-place");

  const auto start_wait = node_->now();
  while (joint_state_msg_count_.load(std::memory_order_relaxed) == 0) {
    if ((node_->now() - start_wait).seconds() > joint_state_wait_timeout_sec_) {
      RCLCPP_WARN(node_->get_logger(), "Task 1: no /joint_states before timeout; continuing");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  auto arm = moveit::planning_interface::MoveGroupInterface(node_, "panda_arm");
  auto hand = moveit::planning_interface::MoveGroupInterface(node_, "hand");

  arm.setPoseReferenceFrame("panda_link0");
  arm.setEndEffectorLink("panda_hand");
  arm.setPlanningTime(5.0);
  arm.setNumPlanningAttempts(10);
  arm.setMaxVelocityScalingFactor(0.2);
  arm.setMaxAccelerationScalingFactor(0.2);
  hand.setPlanningTime(2.0);
  hand.setNumPlanningAttempts(5);

  const double obj_x = request->object_loc.pose.position.x;
  const double obj_y = request->object_loc.pose.position.y;
  const double obj_z = request->object_loc.pose.position.z;
  const double goal_x = request->goal_loc.point.x;
  const double goal_y = request->goal_loc.point.y;
  const double goal_z = request->goal_loc.point.z;

  const double safe_z = obj_z + pick_offset_z_;
  const double grasp_z = obj_z + grasp_approach_offset_z_;
  const double place_safe_z = goal_z + place_offset_z_;
  const double place_z = goal_z + 0.20;

  // 1. Open gripper
  if (!set_gripper(hand, 0.08)) return;

  // 2. Move above object
  if (!move_arm_to_pose(arm, make_pose(obj_x, obj_y, safe_z))) return;

  // 3. Cartesian descend to grasp height
  if (!cartesian_move(arm, make_pose(obj_x, obj_y, grasp_z))) {
    return;
  }

  // 4. Close gripper
  if (!set_gripper(hand, gripper_grasp_width_)) {
    return;
  }

  // 5. Lift object
  if (!cartesian_move(arm, make_pose(obj_x, obj_y, safe_z))) {
    return;
  }

  // 6. Move above basket
  if (!move_arm_to_pose(arm, make_pose(goal_x, goal_y, place_safe_z))) return;

  // 7. Descend to place height
  if (!cartesian_move(arm, make_pose(goal_x, goal_y, place_z))) {
    return;
  }

  // 8. Open gripper to release (0.06 to avoid tolerance issues)
  set_gripper(hand, 0.06);

  // 9. Retreat upward
  if (!move_arm_to_pose(arm, make_pose(goal_x, goal_y, place_safe_z))) {
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "Task 1 callback complete");
}

///////////////////////////////////////////////////////////////////////////////
void
cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{

  if(!latest_cloud_)
  {
    RCLCPP_ERROR(node_->get_logger(),"No pointcloud received");
    return;
  }


  RCLCPP_INFO(node_->get_logger(),"Task2 started");

  auto arm = moveit::planning_interface::MoveGroupInterface(node_, "panda_arm");

  arm.setPoseReferenceFrame("panda_link0");
  arm.setEndEffectorLink("panda_hand");
  arm.setPlanningTime(5.0);

  // ---------- Step1: scan environment ----------
  std::vector<geometry_msgs::msg::Pose> views;

  views.push_back(make_pose(0.4,0.3,0.6));
  views.push_back(make_pose(0.4,0.0,0.6));
  views.push_back(make_pose(0.4,-0.3,0.6));

  for(const auto &v : views)
  {
    move_arm_to_pose(arm,v);
    rclcpp::sleep_for(std::chrono::milliseconds(800));
  }

  // ---------- Step2: check cloud ----------
  if(!latest_cloud_)
  {
    RCLCPP_ERROR(node_->get_logger(),"No cloud received");
    return;
  }
  RCLCPP_INFO(node_->get_logger(),
  "Cloud frame: %s", latest_cloud_->header.frame_id.c_str());

  std::vector<std::string> results;

  for(const auto &loc : request->basket_locs)
  {
    geometry_msgs::msg::PointStamped cam_point;
    geometry_msgs::msg::PointStamped world_point = loc;

    world_point.header.stamp = rclcpp::Time(0);

    try
    {
      cam_point = tf_buffer_.transform(
          world_point,
          latest_cloud_->header.frame_id,
          tf2::durationFromSec(1.0));
          
    }
    catch(tf2::TransformException &ex)
    {
      RCLCPP_WARN(node_->get_logger(),"TF failed: %s", ex.what());
      results.push_back("none");
      continue;
    }

    double target_x = cam_point.point.x;
    double target_y = cam_point.point.y;

    double r_sum=0;
    double g_sum=0;
    double b_sum=0;

    int count=0;

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*latest_cloud_,"x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*latest_cloud_,"y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*latest_cloud_,"z");

    sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_r(*latest_cloud_,"r");
    sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_g(*latest_cloud_,"g");
    sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_b(*latest_cloud_,"b");

    for(; iter_x != iter_x.end();
        ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b)
    {
      double dx = *iter_x - target_x;
      double dy = *iter_y - target_y;
      

      double dist = std::sqrt(dx*dx + dy*dy);

      if(dist < 0.1)
      {
        r_sum += *iter_r;
        g_sum += *iter_g;
        b_sum += *iter_b;

        count++;
      }
    }

    if(count < 8)
    {
      results.push_back("none");
      continue;
    }

    double r = r_sum / count;
    double g = g_sum / count;
    double b = b_sum / count;

    if(r > 150 && b > 150)
      results.push_back("purple");
    else if(r > b)
      results.push_back("red");
    else
      results.push_back("blue");
  }

  response->basket_colours = results;
  for(size_t i = 0; i < results.size(); i++)
  {
    RCLCPP_INFO(node_->get_logger(),
    "Basket %ld at (%.2f, %.2f) -> %s",
    i,
    request->basket_locs[i].point.x,
    request->basket_locs[i].point.y,
    results[i].c_str());
  }

  RCLCPP_INFO(node_->get_logger(),"Task2 finished");

}


///////////////////////////////////////////////////////////////////////////////

void
cw1::t3_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response)
{
  /* function which should solve task 3 */

  (void)request;
  (void)response;
  RCLCPP_INFO_STREAM(
    node_->get_logger(),
    "Task 3 callback triggered (template stub). joint_msgs=" <<
      joint_state_msg_count_.load(std::memory_order_relaxed) <<
      ", cloud_msgs=" << cloud_msg_count_.load(std::memory_order_relaxed));
}
