import uuid
from pydantic import BaseModel, Field, UUID4


class MetaPrompt(BaseModel):
    """
    A meta prompt is a prompt that can be used to generate a prompt for an agent.
    An alternative to a meta-prompt is to allow the agent to ask a human some clarifying questions.
    """

    id: UUID4 = Field(default_factory=uuid.uuid4)
    description: str


class ExplorationGoal(BaseModel):
    goal: str
    examples: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {"goal": self.goal, "examples": self.examples}


class GoalGroup(BaseModel):
    group_name: str
    goals: list[ExplorationGoal]

    def to_dict(self) -> dict[str, list[dict[str, list[str]]]]:
        return {self.group_name: [goal.to_dict() for goal in self.goals]}


class MetaGoal(BaseModel):
    metagoal_description: str
    examples: list[str]


def get_default_meta_goals() -> list[MetaGoal]:
    metagoals = [
        {
            "metagoal": "Identify real-world problems that can be solved by the VMR",
            "examples": [
                "Identify an important market and increase its efficiency through software and AI",
                "Identify an important industry and increase its efficiency through software and AI",
                "Identify a societal problem and solve it through software and AI",
                "Identify a personal problem and solve it through software and AI",
                "Identify an important industry or product and reduce its greenhouse gas emissions through software and AI",
                "Identify an important industry or product and reduce its energy consumption through software and AI",
                "Identify an important industry or product and reduce its water consumption through software and AI",
                "Identify an important industry or product and reduce its waste generation through software and AI",
                "Identify an important industry or product and reduce its pollution through software and AI",
                "Identify an important industry or product and reduce its deforestation through software and AI",
                "Identify an important industry or product and reduce its biodiversity loss through software and AI",
                "Identify an important industry or product and reduce its carbon footprint through software and AI",
            ],
        }
    ]
    return [MetaGoal(**metagoal) for metagoal in metagoals]


def get_default_extrinsic_goals() -> list[GoalGroup]:
    extrinsic_goals = {
        "curiosity": [],
        "novelty": [],
        "self_consistency": [],
        "computational_efficiency": [],
        "quality_goals": [],
        "creative_goals": [],
    }
    return extrinsic_goals


def get_default_intrinsic_goals() -> list[GoalGroup]:
    intrinsic_goals = {
        "curiosity": [
            {
                "goal": "Explore new ideas and concepts",
                "examples": [
                    "Discover and implement new algorithms",
                    "Experiment with different programming paradigms",
                    "Explore the use of new technologies",
                    "Compare different design patterns and choose the most appropriate one",
                    "Evaluate the impact of different design decisions on system maintainability",
                ],
            },
            {
                "goal": "Challenge hidden assumptions",
                "examples": [
                    "Identify and question hidden assumptions in code and design",
                    "Critically evaluate the assumptions made by a system",
                    "Challenge the underlying assumptions of a system",
                    "Identify and test the limits of a system",
                ],
            },
            {
                "goal": "Rephrase problems",
                "examples": [
                    "Reformulate a problem to make it more tractable",
                    "Identify and rephrase implicit assumptions in a problem",
                    "Challenge the validity of a problem statement",
                    "Reformulate a problem to focus on the underlying principles",
                    "Reformulate a problem to make it more specific",
                    "Reformulate a problem to make it more general",
                    "Reformulate a problem to make it more abstract",
                    "Reformulate a problem to make it more concrete",
                ],
            },
            {
                "goal": "Ask follow-up questions",
                "examples": [
                    "Identify and ask follow-up questions to clarify a problem",
                    "Identify and ask follow-up questions to clarify a solution",
                    "Identify and ask follow-up questions to clarify a design decision",
                    "Identify and ask follow-up questions to clarify a code implementation",
                ],
            },
            {
                "goal": "Seek clarification",
                "examples": [
                    "Seek clarification on a problem statement",
                    "Seek clarification on a solution",
                    "Seek clarification on a design decision",
                    "Seek clarification on a code implementation",
                ],
            },
            {
                "goal": "Identify and resolve ambiguities",
                "examples": [
                    "Identify and resolve ambiguities in a problem statement",
                    "Identify and resolve ambiguities in a solution",
                    "Identify and resolve ambiguities in a design decision",
                    "Identify and resolve ambiguities in a code implementation",
                ],
            },
            {
                "goal": "Identify and resolve conflicts",
                "examples": [
                    "Identify and resolve conflicts in a problem statement",
                    "Identify and resolve conflicts in a solution",
                    "Identify and resolve conflicts in a design decision",
                    "Identify and resolve conflicts in a code implementation",
                ],
            },
            {
                "goal": "Identify and resolve inconsistencies",
                "examples": [
                    "Identify and resolve inconsistencies in a problem statement",
                    "Identify and resolve inconsistencies in a solution",
                    "Identify and resolve inconsistencies in a design decision",
                    "Identify and resolve inconsistencies in a code implementation",
                ],
            },
            {
                "goal": "Identify and resolve contradictions",
                "examples": [
                    "Identify and resolve contradictions in a problem statement",
                    "Identify and resolve contradictions in a solution",
                    "Identify and resolve contradictions in a design decision",
                    "Identify and resolve contradictions in a code implementation",
                ],
            },
        ],
        "novelty": [],
        "self_consistency": [],
        "computational_efficiency": [],
        "quality_goals": [
            {
                "goal": "Improve code quality",
                "examples": [
                    "Refactor complex functions",
                    "Increase test coverage",
                    "Implement consistent coding standards",
                    "Reduce code duplication",
                    "Improve error handling and logging",
                    "Enhance documentation and comments",
                    "Optimize resource usage",
                    "Implement design patterns where appropriate",
                    "Add type annotations",
                    "Add function signatures",
                    "Add type aliases",
                    "Add code assertions",
                    "Add preconditions",
                    "Add postconditions",
                    "Add invariants",
                    "Add class and method contracts",
                    "Add design by contract principles",
                    "Add design patterns",
                    "Add design principles. @crawl[max_depth=3](https://en.wikipedia.org/wiki/Category:Programming_principles)",
                    "Add design patterns",
                    "",
                ],
            },
            {
                "goal": "Improve code understandability",
                "examples": [
                    "De-duplicate the code in all VMR repos into a virtual shared code base and the minimize the amount of novel code intrinsic to each repo.",
                    "Use descriptive variable and function names",
                    "Add inline comments for complex logic",
                    "Break down large functions into smaller, focused ones",
                    "Use consistent indentation and formatting",
                    "Implement clear and logical code structure",
                    "Create meaningful abstractions and interfaces",  # TODO: Abstractions emerge from multiple examples or implementations
                    "Use design patterns appropriately",
                    "Write self-documenting code",
                    "Implement clear error messages and logging",
                    "Create comprehensive API documentation",
                    "Use meaningful file and directory names",
                    "Implement consistent naming conventions",
                    "Add explanatory comments for non-obvious code",
                    "Create diagrams for complex workflows",
                    "Write clear and concise commit messages",
                    "Implement modular and reusable code",
                    "Use appropriate data structures",
                    "Implement clear separation of concerns",
                    "Create meaningful test cases with descriptive names",
                    "Document architectural decisions and rationales",
                ],
            },
            {
                "goal": "Optimize performance",
                "examples": ["Improve database query efficiency", "Implement caching"],
            },
            {
                "goal": "Optimize resource utilization",
                "examples": [
                    "Memory management: Reduce memory usage",
                    "Energy management: Improve battery life for mobile apps",
                ],
            },
            {
                "goal": "Enhance user experience",
                "examples": [
                    "Implement responsive design",
                    "Improve accessibility features",
                ],
            },
            {
                "goal": "Automate repetitive tasks",
                "examples": ["Set up CI/CD pipelines", "Create code generation tools"],
            },
            {
                "goal": "Enhance auditability",
                "examples": [
                    "Implement detailed logging",
                    "Create audit trails for critical operations",
                ],
            },
            {
                "goal": "Improve data governance",
                "examples": ["Implement data classification", "Ensure GDPR compliance"],
            },
            {
                "goal": "Enhance system resilience",
                "examples": [
                    "Implement circuit breakers",
                    "Improve error handling and recovery",
                    "Implement self-healing mechanisms",
                    "Implement health checks",
                    "Implement self-stabilization mechanisms",
                    "Implement timeouts",
                    "Implement retries",
                    "Implement fallbacks",
                    "Implement load balancers",
                    "Implement redundancy",
                    "Implement backups",
                    "Implement recovery plans",
                    "Implement recovery protocols",
                    "Implement recovery strategies",
                ],
            },
            {
                "goal": "Optimize cloud resource usage",
                "examples": [
                    "Implement auto-scaling",
                    "Use serverless architectures where appropriate",
                ],
            },
            {
                "goal": "Improve API design and management",
                "examples": [
                    "Implement versioning",
                    "Create comprehensive API documentation",
                ],
            },
            {
                "goal": "Enhance monitoring and observability",
                "examples": [
                    "Implement distributed tracing",
                    "Set up comprehensive dashboards" "Implement monitoring",
                    "Implement logging",
                    "Implement metrics",
                    "Implement alerts",
                    "Implement notifications",
                    "Implement dashboards",
                    "Use observability frameworks like OpenTelemetry, Prometheus, or Grafana",
                ],
            },
            {
                "goal": "Implement advanced analytics capabilities",
                "examples": [
                    "Integrate machine learning models",
                    "Set up data pipelines",
                ],
            },
            {
                "goal": "Improve configuration management",
                "examples": ["Implement feature flags", "Centralize configuration"],
            },
            {
                "goal": "Enhance collaboration features",
                "examples": [
                    "Implement real-time editing",
                    "Improve commenting systems",
                ],
            },
            {
                "goal": "Optimize for edge computing",
                "examples": [
                    "Implement offline capabilities",
                    "Optimize for low-latency operations",
                ],
            },
            {
                "goal": "Enhance sustainability",
                "examples": [
                    "Optimize code for energy efficiency",
                    "Implement green coding practices across repositories",
                    "Develop algorithms for better resource management in cloud environments",
                    "Create tools to measure and reduce the carbon footprint of software",
                ],
            },
            {
                "goal": "Improve accessibility",
                "examples": [
                    "Implement WCAG 2.1 guidelines across all user interfaces",
                    "Develop voice-controlled interfaces for applications",
                    "Create high-contrast and screen reader-friendly versions of UIs",
                    "Implement multi-modal interaction patterns in user interfaces",
                ],
            },
            {
                "goal": "Enhance privacy and data protection",
                "examples": [
                    "Implement privacy by design principles across all repositories",
                    "Develop advanced encryption methods for data at rest and in transit",
                    "Create tools for automated GDPR compliance checking",
                    "Implement zero-knowledge proof systems for authentication",
                ],
            },
            {
                "goal": "Modernize and rejuvenate codebases",
                "examples": [
                    "Identify cases where a major repo-wide refactoring and rewriting is needed",
                    "Replace deprecated APIs with modern alternatives",
                    "Update code to use modern programming paradigms (e.g., async/await, generators)",
                    "Migrate code to use modern programming languages",
                    "Update code to use modern build systems",
                    "Update code to use modern testing frameworks",
                    "Update code to use modern documentation tools",
                    "Update code to use modern version control systems",
                ],
            },
            {
                "goal": "Salvage legacy codebases to recover value and knowledge from obselete codebases",
                "examples": [
                    "Identify and fix critical vulnerabilities in legacy code",
                    "Modernize legacy codebases",
                    "Re-implement legacy codebases",
                    "Refactor legacy codebases",
                    "Update legacy codebases",
                    "Migrate legacy codebases",
                    "Update legacy codebases",
                ],
            },
            {
                "goal": "Automated Program Repair",
                "examples": [
                    "Fix bugs automatically",
                    "Refactor code automatically",
                    "Optimize code automatically",
                    "Update code automatically",
                    "Migrate code automatically",
                    "Re-implement code automatically",
                    "Modernize code automatically",
                    "Salvage code automatically",
                    "Rejuvenate code automatically",
                ],
            },
            {
                "goal": """
                Separate the code base into **mechanism** and **policy** parts
                ([Separation of mechanism and policy](https://en.wikipedia.org/wiki/Separation_of_mechanism_and_policy))
                Mechanisms (those parts of a system implementation that control the authorization of operations and
                the allocation of resources) should not dictate (or overly restrict) the policies according to which
                decisions are made about which operations to authorize, and which resources to allocate.
                Policies do not have to be hardcoded into executable code but can be specified as an independent description.
                """,
                "examples": [
                    "Implement a mechanism that can be used to implement multiple policies",
                    "Implement a policy that can be used with multiple mechanisms",
                    "In a microkernel, the majority of operating system services are provided by user-level server processes.",
                    """In a hotel, use card keys to gain access to locked doors. The mechanisms (magnetic card
                    readers, remote controlled locks, connections to a security server) do not impose any
                    limitations on entrance policy (which people should be allowed to enter which doors,
                    at which times). These decisions are made by a centralized security server, which (in turn)
                    probably makes its decisions by consulting a database of room access rules.""",
                ],
            },
        ],
        "creative_goals": [
            {
                "goal": "Perform design space exploration",
                "examples": [
                    "Explore different ways to implement a feature",
                    "Compare different algorithms for a given task",
                    "Evaluate the trade-offs between different design choices",
                    "Identify potential performance bottlenecks and optimize accordingly",
                    "Explore the use of new technologies or paradigms",
                    "Compare different design patterns and choose the most appropriate one",
                    "Evaluate the impact of different design decisions on system maintainability",
                ],
            },
            {
                "goal": "Use one or more generic problem-solving strategies",
                "examples": [
                    "Abstraction: solving the problem in a tractable model system to gain insight into the real system",
                    "Analogy: adapting the solution to a previous problem which has similar features or mechanisms",
                    "Brainstorming: suggesting a large number of solutions or ideas and combining and developing them until an optimum solution is found",
                    "Bypasses: transform the problem into another problem that is easier to solve, bypassing the barrier, then transform that solution back to a solution to the original problem.",
                    "Critical thinking: analysis of available evidence and arguments to form a judgement via rational, skeptical, and unbiased evaluation",
                    "Divide and conquer: breaking down a large, complex problem into smaller, solvable problems",
                    "Help-seeking: obtaining external assistance (e.g., from tools) to deal with obstacles",
                    "Hypothesis testing: assuming a possible explanation to the problem and trying to prove (or, in some contexts, disprove) the assumption",
                    "Lateral thinking: approaching solutions indirectly and creatively",
                    "Means-ends analysis: choosing an action at each step to move closer to the goal",
                    "Morphological analysis: assessing the output and interactions of an entire system",
                    "Proof of impossibility: try to prove that the problem cannot be solved. The point where the proof fails will be the starting point for solving it",
                    "Reduction: transforming the problem into another problem for which solutions exist",
                    "Research: employing existing ideas or adapting existing solutions to similar problems",
                    "Root cause analysis: identifying the cause of a problem",
                    "Trial-and-error: testing possible solutions until the right one is found",
                ],
            },
            {
                "goal": "Challenge hidden assumptions",
                "examples": [
                    "",
                ],
            },
            {
                "goal": "Rephrase problems",
                "examples": [
                    "",
                ],
            },
            {
                "goal": "Ask follow-up questions",
                "examples": [
                    "",
                ],
            },
            {
                "goal": "Make new connections between exisitng concepts and ideas",
                "examples": [
                    "",
                ],
            },
            {
                "goal": "Explore new paradigms",
                "examples": [
                    "Implement a functional reactive programming paradigm",
                    "Implement a reactive programming paradigm",
                    "Implement a declarative programming paradigm",
                ],
            },
            {
                "goal": "Implement new features",
                "examples": ["Add user authentication", "Integrate a new API"],
            },
            {
                "goal": "Break language barriers",
                "examples": [
                    "Create Python bindings for a C++ library and apply them in novel contexts",
                    "Use WebAssembly to run Rust code in a web app",
                    "Develop Java Native Interface (JNI) to call C/C++ code from Java applications",
                    "Utilize Foreign Function Interface (FFI) to call Rust functions from Python",
                    "Implement JavaScript bindings for a Go library using cgo and WebAssembly",
                    "Create Ruby extensions using C to optimize performance-critical parts of an application",
                    "Use Cython to speed up Python code by compiling it to C",
                    "Develop .NET interop to call native C++ libraries from C# applications",
                ],
            },
            # It is not about bridging projects written in different languages, but more importantly
            # about enabling novel use cases that are low-hanging fruits by integrating multiple separately
            # developed technologies.
            {
                "goal": "Exploit multi-repo integration opportunities",
                "examples": [
                    "Enable desktop apps in JavaScript by integrating Node.js with Chromium",
                    "Create a real-time collaborative coding environment by combining a text editor with WebRTC",
                    "Build a voice-controlled smart home system by integrating speech recognition with IoT device APIs",
                    "Develop an augmented reality shopping experience by combining computer vision with e-commerce platforms",
                    "Create a personalized health dashboard by integrating wearable device data with medical knowledge bases",
                    "Build a multi-language code translator by combining abstract syntax tree parsing with machine translation",
                    "Develop an AI-powered game level generator by integrating procedural content generation with machine learning",
                    "Create a privacy-focused decentralized social network by combining blockchain technology with peer-to-peer networking",
                ],
            },
            {
                "goal": "Enable new business models",
                "examples": [
                    "Implement a subscription system",
                    "Create a marketplace for plugins",
                ],
            },
            {
                "goal": "Transfer learnings across repositories perhaps across language barriers",
                "examples": [
                    "Apply successful testing strategies from a Python project to a JavaScript project"
                ],
            },
            {
                "goal": "Explore emerging technologies",
                "examples": [
                    "Implement quantum computing algorithms in a classical system",
                    "Develop a proof-of-concept for edge computing in IoT devices",
                    "Create a prototype using neuromorphic computing principles",
                    "Experiment with federated learning in a distributed system",
                ],
            },
            {
                "goal": "Buidl multi-domain, multi-modal apps with multiple deep-tech stacks to address complex real-world problems and undo the damage of hyper-specialization",
                "examples": [
                    """
                    Use low-fidelity simulation that can quickly discover **highly consequential scenarios** for high-fidelity simulators in financial markets, wars, business planning, game playing, urban planning."""
                    "Address problems of very long and interconnected supply chains, markets and industries"
                    "Create a system that can detect and respond to multiple types of cyber attacks",
                    "Develop a solution that can optimize resource allocation across multiple departments",
                    "Implement a system that can predict and respond to multiple types of natural disasters",
                ],
            },
        ],
        "innovation_goals": [],
        "developer_experience_goals": [
            {
                "goal": "Build a knowledge browser for codebases in the VMR that offers a rich, interactive, and intuitive cognitive map to navigate and explore codebases and discover hidden connections and patterns and new extensions.",
                "examples": [
                    "Implement a code search engine that indexes all repositories in the VMR",
                    "Develop a code recommendation system that suggests references to relevant code locations in the VMR based on context",
                    "Create a code visualization tool that displays code relationships and dependencies",
                    "Build a code review tool that highlights potential issues and suggests improvements",
                    "Implement a code metrics dashboard that provides insights into code quality and performance",
                    "Develop a code refactoring tool that automates common code improvements",
                    "Create a code annotation tool that adds comments and documentation to code",
                    "Implement a code testing framework that automates testing and validation",
                    "Develop a code deployment tool that streamlines the release process",
                    "Create a code monitoring tool that tracks code changes and performance metrics",
                    "Build a code profiling tool that identifies performance bottlenecks",
                    "Implement a code versioning system that tracks changes and enables rollbacks",
                    "Develop a code documentation generator that creates API documentation and user guides",
                    "Create a code optimization tool that improves resource usage and performance",
                    "Build a code transformation tool that converts code between languages and frameworks",
                    "Implement a code analysis tool that identifies bugs and security vulnerabilities",
                ],
            },
            {
                "goal": "Enhance developer experience",
                "examples": [
                    "Visualize or animate an abstract view of the build-time and runtime execution flow.",
                    "Standardize build processes",
                    "Improve documentation",
                    "Implement consistent code formatting across repositories",
                    "Create unified development environments using containers",
                    "Develop shared libraries and tools for common tasks",
                    "Develop repository-specific tools to help with coding and development tasks"
                    "Implement automated code review processes",
                    "Establish clear contribution guidelines",
                    "Set up centralized package management",
                    "Create interactive onboarding tutorials for new developers",
                    "Implement automated testing frameworks across all repositories",
                    "Develop a comprehensive code quality dashboard for all repositories on GitHub or GitLab",
                ],
            },
        ],
        "portability_goals": [
            {
                "goal": "Ensure cross-platform compatibility",
                "examples": [
                    "Ensure mobile apps work on both iOS and Android",
                    "Develop a cross-platform GUI framework that works on Windows, macOS, and Linux",
                    "Create a mobile app that runs on both iOS and Android using a single codebase",
                    "Implement a web application that functions consistently across different browsers",
                ],
            },
            {
                "goal": "Enable easy deployment across different environments",
                "examples": [
                    "Containerize applications for consistent deployment across development, testing, and production environments",
                    "Create platform-independent build scripts that work on different CI/CD systems",
                    "Develop a configuration management system that adapts to various cloud providers",
                ],
            },
            {
                "goal": "Ensure data portability and interoperability",
                "examples": [
                    "Implement standard data exchange formats (e.g., JSON, XML) for all APIs",
                    "Create data migration tools to move between different database systems",
                    "Develop plugins for popular software to import/export data in various formats",
                ],
            },
            {
                "goal": "Implement cross-language interoperability",
                "examples": [
                    "Develop language bindings to use a library written in one language from another language",
                    "Create a REST API to expose functionality across different programming languages",
                    "Implement a message queue system for communication between services written in different languages",
                ],
            },
            {
                "goal": "Ensure backward compatibility",
                "examples": [
                    "Maintain compatibility with older versions of APIs",
                    "Implement versioning for APIs to support older clients",
                    "Develop strategies for rolling back changes in case of compatibility issues",
                ],
            },
            {
                "goal": "Enable easy migration between platforms",
                "examples": [
                    "Develop tools to convert data between different file formats",
                    "Create scripts to migrate data between cloud providers",
                    "Implement a strategy to move applications between on-premises and cloud environments",
                ],
            },
            {
                "goal": "Generate platform-independent code",
                "examples": [
                    "Use cross-platform libraries and frameworks",
                    "Develop code that adheres to open standards",
                    "Implement platform-independent data formats",
                    "Create platform-independent APIs",
                ],
            },
        ],
        "sustainability_goals": [],
        "user_experience_goals": [],
        "scalability_goals": [],
        "security_goals": [
            {
                "goal": "Identify and mitigate systemic vulnerabilities",
                "examples": [
                    "An arbitrary code execution vulnerability in a web application triggered by a benign user input that passes input validation but gets transformed by other layers of the system into a malicious payload",
                ],
            },
            {
                "goal": "Discover and address security vulnerabilities",
                "examples": [
                    "Implement automated vulnerability scanning across all repositories",
                    "Conduct regular penetration testing to identify potential exploits",
                    "Develop a process for responsible disclosure of discovered vulnerabilities",
                    "Create a security bug bounty program to encourage external security research",
                ],
            },
            {
                "goal": "Explore and mitigate emerging security threats",
                "examples": [
                    "Research and implement defenses against zero-day exploits",
                    "Develop strategies to combat AI-powered cyber attacks",
                    "Investigate quantum-resistant cryptographic algorithms",
                    "Create simulations of advanced persistent threats (APTs) for testing defenses",
                ],
            },
            {
                "goal": "Enhance security through code analysis",
                "examples": [
                    "Implement static code analysis tools to detect security flaws",
                    "Develop dynamic analysis techniques for runtime security checks",
                    "Create machine learning models to identify potential security vulnerabilities in code",
                    "Implement automated security testing in the CI/CD pipeline",
                ],
            },
            {
                "goal": "Propose new security features",
                "examples": [
                    "Implement a new authentication method",
                    "Integrate a new security library",
                ],
            },
            {
                "goal": "Propose new exploits of discovered vulnerabilities",
                "examples": [
                    "Create a simulation of a zero-day exploit",
                    "Develop a proof-of-concept for a new type of cyber attack",
                ],
            },
        ],
        "accessibility_goals": [],
        "internationalization_goals": [],
        "compliance_goals": [],
    }

    intrinsic_goals = [
        GoalGroup(group_name=group_name, goals=[ExplorationGoal(**goal) for goal in goals])
        for group_name, goals in intrinsic_goals.items()
    ]
    return intrinsic_goals

