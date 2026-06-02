std_msgs/Header header

# In-process trace of a Cortex planning/execution episode.
# The generic recorder captures this topic without interpreting it;
# the offline dataset builders turn these traces into Cortex datasets.

# event_type values (also the episode-boundary signal: TASK_RECEIVED opens,
# EPISODE_COMPLETE / EPISODE_ABORTED close)
string TASK_RECEIVED="TASK_RECEIVED"
string PLAN_GENERATED="PLAN_GENERATED"
string CONFIRMATION="CONFIRMATION"
string STEP_EXECUTING="STEP_EXECUTING"
string STEP_COMPLETED="STEP_COMPLETED"
string STEP_FAILED="STEP_FAILED"
string EPISODE_COMPLETE="EPISODE_COMPLETE"
string EPISODE_ABORTED="EPISODE_ABORTED"

string episode_id         # uuid per task goal
string component_node     # emitting Cortex node name
string event_type         # one of the constants above
uint32 step_index         # plan step this event belongs to (0 for episode-level events)
string task               # populated only on TASK_RECEIVED
string tool_name          # populated for step-level events
string payload_json       # variant data: messages, tool_calls, tools catalog, results
