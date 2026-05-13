# TradingAgents — UML Architecture Diagrams

All diagrams are written in [PlantUML](https://plantuml.com/) format (`.puml`).

## How to Render

**Option 1 — VS Code extension**
Install [PlantUML](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml) extension, then `Alt+D` to preview.

**Option 2 — CLI (requires Java + Graphviz)**
```bash
java -jar plantuml.jar docs/uml/*.puml
```

**Option 3 — Online**
Paste `.puml` content into [plantuml.com/plantuml](https://www.plantuml.com/plantuml/uml/) or [kroki.io](https://kroki.io/).

**Option 4 — Docker**
```bash
docker run --rm -v $(pwd):/data plantuml/plantuml -tpng /data/docs/uml/*.puml
```

---

## Diagram Index

| # | File | UML Type | What it shows |
|---|------|----------|---------------|
| 01 | [01_class_diagram_agents.puml](01_class_diagram_agents.puml) | **Class** | Agent hierarchy, LLM clients, state models, structured schemas |
| 02 | [02_class_diagram_dataflows.puml](02_class_diagram_dataflows.puml) | **Class** | Dataflow vendor abstraction, tool routing, yfinance vs Alpha Vantage |
| 03 | [03_sequence_propagate.puml](03_sequence_propagate.puml) | **Sequence** | Full `propagate()` call — from user input to final decision |
| 04 | [04_sequence_debate.puml](04_sequence_debate.puml) | **Sequence** | Investment debate (Bull↔Bear→RM) + Risk debate (Agg→Con→Neu→PM) |
| 05 | [05_sequence_memory_reflection.puml](05_sequence_memory_reflection.puml) | **Sequence** | Memory lifecycle: store PENDING, resolve outcomes, reflect, inject context |
| 06 | [06_component_diagram.puml](06_component_diagram.puml) | **Component** | All system components and their inter-dependencies |
| 07 | [07_package_diagram.puml](07_package_diagram.puml) | **Package** | Python package structure and import dependencies |
| 08 | [08_use_case_diagram.puml](08_use_case_diagram.puml) | **Use Case** | User interactions, actor roles, include/extend relationships |
| 09 | [09_state_diagram_debate.puml](09_state_diagram_debate.puml) | **State** | Investment debate, risk debate, and analyst tool-loop state machines |
| 10 | [10_state_diagram_graph_execution.puml](10_state_diagram_graph_execution.puml) | **State** | Full graph execution lifecycle including checkpoint and memory states |
| 11 | [11_activity_diagram_full_pipeline.puml](11_activity_diagram_full_pipeline.puml) | **Activity** | End-to-end pipeline activity flow with swimlanes per agent |
| 12 | [12_activity_diagram_tool_routing.puml](12_activity_diagram_tool_routing.puml) | **Activity** | Dataflow vendor routing logic (tool-level → category → default) |
| 13 | [13_deployment_diagram.puml](13_deployment_diagram.puml) | **Deployment** | Runtime deployment: local machine, LLM clouds, market data APIs |
| 14 | [14_object_diagram.puml](14_object_diagram.puml) | **Object** | Concrete runtime snapshot of AgentState for AAPL 2025-01-10 |
| 15 | [15_sequence_llm_factory.puml](15_sequence_llm_factory.puml) | **Sequence** | LLM client factory, provider selection, structured output flow |
| 16 | [16_communication_diagram.puml](16_communication_diagram.puml) | **Communication** | Which agents read/write which state fields — data handoff map |
| 17 | [17_timing_diagram.puml](17_timing_diagram.puml) | **Timing** | Approximate relative execution timeline across all pipeline phases |
| 18 | [18_sequence_cli.puml](18_sequence_cli.puml) | **Sequence** | CLI user interaction flow with live Rich console output |
