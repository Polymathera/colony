from pydantic import BaseModel, Field


class ExplorationConstraints(BaseModel):
    time_limit: int = 24 * 60 * 60  # 24 hours in seconds
    max_depth: int = 5  # Maximum depth of exploration
    max_branches: int = 10  # Maximum number of branches to explore at each level
    max_cumulative_prompt_tokens: int = 1000_000_000
    max_cumulative_completion_tokens: int = 1000_000_000
    max_cumulative_total_tokens: int = 1000_000_000
    max_in_step_prompt_tokens: int = 100_000
    max_in_step_completion_token: int = 100_000
    max_in_step_total_tokens: int = 100_000
    resource_limit: dict[str, int] = Field(default_factory=dict)
    allowed_technologies: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)
    priority_areas: list[str] = Field(default_factory=list)
    ethical_constraints: list[str] = Field(default_factory=list)


def get_default_exploration_constraints() -> ExplorationConstraints:
    return ExplorationConstraints(
        time_limit=24 * 60 * 60,  # 24 hours in seconds
        max_depth=5,  # Maximum depth of exploration
        max_branches=10,  # Maximum number of branches to explore at each level
        max_cumulative_prompt_tokens=1000_000_000,
        max_cumulative_completion_tokens=1000_000_000,
        max_cumulative_total_tokens=1000_000_000,
        max_in_step_prompt_tokens=100_000,
        max_in_step_completion_token=100_000,
        max_in_step_total_tokens=100_000,
        resource_limit={
            "cpu": 80,  # Maximum CPU usage percentage
            "memory": 8 * 1024 * 1024 * 1024,  # 8 GB in bytes
            "api_calls": 1000,  # Maximum number of API calls
        },
        allowed_technologies=["Python", "JavaScript", "Java", "C++", "Rust"],
        excluded_paths=["tests/", "docs/", "legacy/"],
        priority_areas=["security", "performance", "scalability"],
        ethical_constraints=[
            "respect_user_privacy",
            "avoid_biased_algorithms",
            "ensure_data_protection",
        ],
    )
