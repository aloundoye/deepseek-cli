use crate::mode_router::{AgentMode, FailureTracker, ModeRouterConfig};
use crate::plan_discipline::PlanState;
use deepseek_core::{ChatMessage, ToolDefinition};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ChatLoopState {
    pub(crate) turn_count: u64,
    pub(crate) failure_streak: u32,
    pub(crate) empty_response_count: u32,
    pub(crate) tool_choice_retried: bool,
    pub(crate) plan_mode_active: bool,
    pub(crate) budget_warned: bool,
    pub(crate) active_tools: Vec<ToolDefinition>,
    pub(crate) all_tools: Vec<ToolDefinition>,
    pub(crate) extended_tools: Vec<ToolDefinition>,
    pub(crate) mode_router_config: ModeRouterConfig,
    pub(crate) failure_tracker: FailureTracker,
    pub(crate) current_mode: AgentMode,
}

impl ChatLoopState {
    #[allow(dead_code)]
    pub(crate) fn new(
        plan_mode_active: bool,
        active_tools: Vec<ToolDefinition>,
        all_tools: Vec<ToolDefinition>,
        extended_tools: Vec<ToolDefinition>,
        mode_router_config: ModeRouterConfig,
    ) -> Self {
        Self {
            turn_count: 0,
            failure_streak: 0,
            empty_response_count: 0,
            tool_choice_retried: false,
            plan_mode_active,
            budget_warned: false,
            active_tools,
            all_tools,
            extended_tools,
            mode_router_config,
            failure_tracker: FailureTracker::default(),
            current_mode: AgentMode::V3Autopilot,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ConversationState {
    pub(crate) messages: Vec<ChatMessage>,
    pub(crate) plan_state: PlanState,
}

impl ConversationState {
    #[allow(dead_code)]
    pub(crate) fn new(messages: Vec<ChatMessage>, plan_state: PlanState) -> Self {
        Self {
            messages,
            plan_state,
        }
    }
}
