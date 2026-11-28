from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import crew_runner, tasks as task_builders
from src.models import (
    ClarificationAnalysis,
    ClarificationQuestion,
    RequirementsAnalysis,
    RequirementsValidation,
    SDLCAnalysis,
    ValidationFinding,
)


class AgentWorkflowTests(unittest.TestCase):
    def test_clarification_and_validation_models_capture_review_output(self):
        clarification = ClarificationAnalysis(
            summary="The transaction threshold is not specified.",
            questions=[
                ClarificationQuestion(
                    question="What transaction amount should trigger review?",
                    reason="The threshold determines when manual review is required.",
                    related_requirement_ids=["FR-01"],
                    priority="High",
                )
            ],
        )
        validation = RequirementsValidation(
            status="pass",
            summary="One requirement lacks a measurable threshold.",
            reviewed_requirement_ids=["FR-01"],
            findings=[
                ValidationFinding(
                    requirement_id="FR-01",
                    issue_type="incomplete",
                    severity="High",
                    description="The review threshold is unspecified.",
                    evidence="The requirement does not define an amount.",
                    recommended_action="Ask the business owner to define the threshold.",
                )
            ],
        )

        self.assertEqual(clarification.model_dump()["questions"][0]["related_requirement_ids"], ["FR-01"])
        self.assertEqual(validation.model_dump()["status"], "needs_review")
        self.assertEqual(validation.model_dump()["findings"][0]["issue_type"], "incomplete")

    def test_agent_tasks_receive_the_prior_outputs_as_context(self):
        requirements_task = object()
        clarification_task = object()

        with patch(
            "src.tasks.Task",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            clarification = task_builders.build_clarification_task(
                "clarification-agent",
                requirements_task,
                "Payment workflow",
                "Source excerpt",
            )
            validation = task_builders.build_validation_task(
                "validation-agent",
                requirements_task,
                clarification_task,
                "Source excerpt",
            )
            sdlc = task_builders.build_sdlc_task(
                "sdlc-agent",
                requirements_task,
                validation,
            )

        self.assertEqual(clarification.context, [requirements_task])
        self.assertEqual(validation.context, [requirements_task, clarification_task])
        self.assertEqual(sdlc.context, [requirements_task, validation])

    def test_run_crew_executes_clarification_and_validation_before_sdlc(self):
        requirements = RequirementsAnalysis(summary="Payment processing.")
        clarification = ClarificationAnalysis(summary="A limit is missing.")
        validation = RequirementsValidation(
            status="needs_review",
            summary="A limit needs stakeholder confirmation.",
        )
        sdlc = SDLCAnalysis(
            recommended_approach="Agile-V-Model hybrid",
            workflow_notes="Use security and compliance gates.",
        )
        task_outputs = [
            SimpleNamespace(output=SimpleNamespace(pydantic=requirements), name="requirements"),
            SimpleNamespace(output=SimpleNamespace(pydantic=clarification), name="clarification"),
            SimpleNamespace(output=SimpleNamespace(pydantic=validation), name="validation"),
            SimpleNamespace(output=SimpleNamespace(pydantic=sdlc), name="sdlc"),
        ]
        tasks = SimpleNamespace(
            build_requirements_task=patch("src.tasks.build_requirements_task", return_value=task_outputs[0]),
            build_clarification_task=patch("src.tasks.build_clarification_task", return_value=task_outputs[1]),
            build_validation_task=patch("src.tasks.build_validation_task", return_value=task_outputs[2]),
            build_sdlc_task=patch("src.tasks.build_sdlc_task", return_value=task_outputs[3]),
        )
        captured = {}

        class FakeCrew:
            def __init__(self, agents, tasks, process, verbose):
                captured.update(agents=agents, tasks=tasks, process=process)

            def kickoff(self):
                return None

        with (
            tasks.build_requirements_task,
            tasks.build_clarification_task,
            tasks.build_validation_task,
            tasks.build_sdlc_task,
            patch("src.agents.build_llm", return_value=object()),
            patch("src.agents.build_requirements_agent", return_value="requirements-agent"),
            patch("src.agents.build_clarification_agent", return_value="clarification-agent"),
            patch("src.agents.build_validation_agent", return_value="validation-agent"),
            patch("src.agents.build_sdlc_agent", return_value="sdlc-agent"),
            patch("src.crew_runner.load_source_documents", return_value=[]),
            patch("src.crew_runner.retrieve_context", return_value=("reference evidence", [])),
            patch(
                "src.crew_runner.build_extended_analysis",
                return_value={"input_governance": {"masking_events": []}},
            ),
            patch("src.crew_runner._save_audit_record", return_value={}),
            patch("crewai.Crew", FakeCrew),
        ):
            result = crew_runner.run_crew(
                scenario="A payment workflow",
                reference_documents=[],
                model_name="test-model",
            )

        self.assertEqual(
            [agent for agent in captured["agents"]],
            ["requirements-agent", "clarification-agent", "validation-agent", "sdlc-agent"],
        )
        self.assertEqual(
            [task.name for task in captured["tasks"]],
            ["requirements", "clarification", "validation", "sdlc"],
        )
        self.assertEqual(
            result.extended_analysis["clarification_agent"],
            clarification.model_dump(),
        )
        self.assertEqual(
            result.extended_analysis["validation_agent"],
            validation.model_dump(),
        )


if __name__ == "__main__":
    unittest.main()
