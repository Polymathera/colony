---
name: ${name}
description: ${description}
---

# ${name}

${description}

## Inputs

Document the inputs the planner should pass when calling this skill via
``UserPluginCapability.run_skill("${name}", ...)``.

## Behaviour

Describe what the skill does. The framework executes this block in a
container governed by the ``sandbox_image`` recorded in the optional
``plugin.json`` sibling — author it through L1-E's ``bootstrap_plugin``
when the skill needs custom runtime dependencies.

## Outputs

Describe what the skill returns.
