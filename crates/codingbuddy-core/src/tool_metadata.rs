use crate::{TaskPhase, ToolName};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolTier {
    Core,
    Contextual,
    Extended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolAgentRole {
    Build,
    Explore,
    Plan,
    Bash,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolPhaseAccess {
    pub explore: bool,
    pub plan: bool,
    pub execute: bool,
    pub verify: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolMetadata {
    pub read_only: bool,
    pub phase_access: ToolPhaseAccess,
    pub agent_level: bool,
    pub review_blocked: bool,
    pub tier: ToolTier,
    pub allowed_roles: &'static [ToolAgentRole],
}

const ALL_ROLES: &[ToolAgentRole] = &[
    ToolAgentRole::Build,
    ToolAgentRole::Explore,
    ToolAgentRole::Plan,
    ToolAgentRole::Bash,
    ToolAgentRole::General,
];
const BUILD_GENERAL: &[ToolAgentRole] = &[ToolAgentRole::Build, ToolAgentRole::General];
const BUILD_PLAN_GENERAL: &[ToolAgentRole] = &[
    ToolAgentRole::Build,
    ToolAgentRole::Plan,
    ToolAgentRole::General,
];
const BUILD_BASH_GENERAL: &[ToolAgentRole] = &[
    ToolAgentRole::Build,
    ToolAgentRole::Bash,
    ToolAgentRole::General,
];
const BUILD_PLAN_BASH_GENERAL: &[ToolAgentRole] = &[
    ToolAgentRole::Build,
    ToolAgentRole::Plan,
    ToolAgentRole::Bash,
    ToolAgentRole::General,
];
const WEB_ROLES: &[ToolAgentRole] = &[
    ToolAgentRole::Explore,
    ToolAgentRole::Plan,
    ToolAgentRole::General,
];
const NOTEBOOK_READ_ROLES: &[ToolAgentRole] = &[
    ToolAgentRole::Build,
    ToolAgentRole::Explore,
    ToolAgentRole::Plan,
    ToolAgentRole::General,
];

const READ_ALL_PHASES: ToolPhaseAccess = ToolPhaseAccess {
    explore: true,
    plan: true,
    execute: true,
    verify: true,
};
const EXECUTE_ONLY: ToolPhaseAccess = ToolPhaseAccess {
    explore: false,
    plan: false,
    execute: true,
    verify: false,
};
const EXECUTE_AND_VERIFY: ToolPhaseAccess = ToolPhaseAccess {
    explore: false,
    plan: false,
    execute: true,
    verify: true,
};
const PLAN_EXECUTE_VERIFY: ToolPhaseAccess = ToolPhaseAccess {
    explore: false,
    plan: true,
    execute: true,
    verify: true,
};
const EXPLORE_EXECUTE_VERIFY: ToolPhaseAccess = ToolPhaseAccess {
    explore: true,
    plan: false,
    execute: true,
    verify: true,
};
const PLAN_ONLY: ToolPhaseAccess = ToolPhaseAccess {
    explore: false,
    plan: true,
    execute: false,
    verify: false,
};

impl ToolName {
    #[must_use]
    pub fn metadata(&self) -> ToolMetadata {
        match self {
            Self::FsRead
            | Self::FsList
            | Self::FsGlob
            | Self::FsGrep
            | Self::GitStatus
            | Self::GitDiff
            | Self::GitShow
            | Self::IndexQuery
            | Self::DiagnosticsCheck
            | Self::Batch
            | Self::UserQuestion
            | Self::TaskGet
            | Self::TaskList
            | Self::TodoRead
            | Self::TaskOutput
            | Self::ExtendedThinking
            | Self::ToolSearch => ToolMetadata {
                read_only: true,
                phase_access: READ_ALL_PHASES,
                agent_level: matches!(
                    self,
                    Self::UserQuestion
                        | Self::TaskGet
                        | Self::TaskList
                        | Self::TodoRead
                        | Self::TaskOutput
                        | Self::ExtendedThinking
                        | Self::ToolSearch
                ),
                review_blocked: false,
                tier: match self {
                    Self::ExtendedThinking => ToolTier::Contextual,
                    Self::ToolSearch => ToolTier::Core,
                    Self::GitStatus | Self::GitDiff | Self::GitShow => ToolTier::Contextual,
                    Self::IndexQuery | Self::DiagnosticsCheck => ToolTier::Contextual,
                    _ => ToolTier::Core,
                },
                allowed_roles: ALL_ROLES,
            },
            Self::WebFetch | Self::WebSearch => ToolMetadata {
                read_only: true,
                phase_access: READ_ALL_PHASES,
                agent_level: false,
                review_blocked: false,
                tier: ToolTier::Contextual,
                allowed_roles: WEB_ROLES,
            },
            Self::NotebookRead => ToolMetadata {
                read_only: true,
                phase_access: READ_ALL_PHASES,
                agent_level: false,
                review_blocked: false,
                tier: ToolTier::Extended,
                allowed_roles: NOTEBOOK_READ_ROLES,
            },
            Self::FsWrite
            | Self::FsEdit
            | Self::MultiEdit
            | Self::PatchStage
            | Self::PatchApply
            | Self::PatchDirect
            | Self::NotebookEdit => ToolMetadata {
                read_only: false,
                phase_access: EXECUTE_ONLY,
                agent_level: false,
                review_blocked: true,
                tier: match self {
                    Self::FsWrite | Self::FsEdit | Self::MultiEdit => ToolTier::Core,
                    _ => ToolTier::Extended,
                },
                allowed_roles: BUILD_GENERAL,
            },
            Self::BashRun => ToolMetadata {
                read_only: false,
                phase_access: EXECUTE_AND_VERIFY,
                agent_level: false,
                review_blocked: true,
                tier: ToolTier::Core,
                allowed_roles: BUILD_BASH_GENERAL,
            },
            Self::ChromeNavigate
            | Self::ChromeClick
            | Self::ChromeTypeText
            | Self::ChromeScreenshot
            | Self::ChromeReadConsole
            | Self::ChromeEvaluate => ToolMetadata {
                read_only: false,
                phase_access: EXECUTE_AND_VERIFY,
                agent_level: false,
                review_blocked: false,
                tier: ToolTier::Extended,
                allowed_roles: BUILD_GENERAL,
            },
            Self::TaskCreate | Self::TaskUpdate | Self::TodoWrite | Self::TaskStop => {
                ToolMetadata {
                    read_only: false,
                    phase_access: PLAN_EXECUTE_VERIFY,
                    agent_level: true,
                    review_blocked: false,
                    tier: ToolTier::Contextual,
                    allowed_roles: BUILD_PLAN_BASH_GENERAL,
                }
            }
            Self::SpawnTask => ToolMetadata {
                read_only: false,
                phase_access: PLAN_EXECUTE_VERIFY,
                agent_level: true,
                review_blocked: false,
                tier: ToolTier::Contextual,
                allowed_roles: BUILD_PLAN_GENERAL,
            },
            Self::EnterPlanMode => ToolMetadata {
                read_only: false,
                phase_access: EXPLORE_EXECUTE_VERIFY,
                agent_level: true,
                review_blocked: false,
                tier: ToolTier::Contextual,
                allowed_roles: BUILD_PLAN_GENERAL,
            },
            Self::ExitPlanMode => ToolMetadata {
                read_only: false,
                phase_access: PLAN_ONLY,
                agent_level: true,
                review_blocked: false,
                tier: ToolTier::Contextual,
                allowed_roles: BUILD_PLAN_GENERAL,
            },
            Self::Skill => ToolMetadata {
                read_only: false,
                phase_access: EXECUTE_AND_VERIFY,
                agent_level: true,
                review_blocked: false,
                tier: ToolTier::Extended,
                allowed_roles: BUILD_GENERAL,
            },
            Self::KillShell => ToolMetadata {
                read_only: false,
                phase_access: EXECUTE_AND_VERIFY,
                agent_level: true,
                review_blocked: false,
                tier: ToolTier::Extended,
                allowed_roles: BUILD_BASH_GENERAL,
            },
        }
    }

    #[must_use]
    pub fn is_allowed_for_role(&self, role: ToolAgentRole) -> bool {
        self.metadata().allowed_roles.contains(&role)
    }

    #[must_use]
    pub fn is_allowed_in_phase(&self, phase: TaskPhase) -> bool {
        let access = self.metadata().phase_access;
        match phase {
            TaskPhase::Explore => access.explore,
            TaskPhase::Plan => access.plan,
            TaskPhase::Execute => access.execute,
            TaskPhase::Verify => access.verify,
        }
    }

    #[must_use]
    pub fn tier(&self) -> ToolTier {
        self.metadata().tier
    }
}

#[must_use]
pub fn is_api_tool_name_read_only(name: &str) -> bool {
    ToolName::from_api_name(name)
        .map(|tool| tool.is_read_only())
        .unwrap_or(false)
}

#[must_use]
pub fn is_internal_tool_name_read_only(name: &str) -> bool {
    ToolName::from_internal_name(name)
        .map(|tool| tool.is_read_only())
        .unwrap_or(false)
}
