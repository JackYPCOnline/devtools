#!/usr/bin/env python3
"""
Strands GitHub Agent Runner
A portable agent runner for use in GitHub Actions across different repositories.
"""

import base64
import json
import os
import sys
from datetime import datetime
from typing import Any

import boto3
from strands import Agent
from strands.telemetry import StrandsTelemetry
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session import S3SessionManager
from strands.models import BedrockModel, CacheConfig
from strands.vended_plugins.context_offloader import ContextOffloader, S3Storage
from botocore.config import Config

from strands_tools import http_request, shell

# Import local GitHub tools we need
from github_tools import (
    add_issue_comment,
    add_issue_labels,
    add_pr_comment,
    create_issue,
    create_pull_request,
    get_invoked_write_tools,
    get_issue,
    get_issue_comments,
    get_issue_label_names,
    get_pull_request,
    get_pr_files,
    get_pr_review_and_comments,
    list_issues,
    list_pull_requests,
    reply_to_review_comment,
    update_issue,
    update_pull_request,
)

# Import local tools we need
from handoff_to_user import handoff_to_user
from notebook import notebook
from str_replace_based_edit_tool import str_replace_based_edit_tool

# Strands configuration constants
STRANDS_MODEL_ID = "global.anthropic.claude-opus-4-8"
STRANDS_MAX_TOKENS = 64000
STRANDS_REGION = "us-west-2"

# Default values for environment variables used only in this file
DEFAULT_SYSTEM_PROMPT = "You are an autonomous GitHub agent powered by Strands Agents SDK."


# Write tools that touch the issue itself. A bug-verifier run must end up applying
# at least one of these (a triage label, or a comment) -- the SOP guarantees a
# label on every verdict, plus a comment in the derived-repro / cannot-reproduce
# cases.
_ISSUE_WRITE_TOOLS = {"add_issue_labels", "add_issue_comment"}

# Triage labels the bug-verifier applies. Used to recognise that a prior run's
# work already landed on the issue when the current (resumed) run wrote nothing.
_TRIAGE_LABELS = {"bug-validated", "bug-needs-info", "bug-cannot-reproduce"}


def _issue_number_from_session_id(session_id: str) -> int | None:
    """Extract the trailing issue number from an issue-scoped session ID.

    Session IDs are formed as "<mode>-<issue>" (e.g. "bug-verifier-3216").
    """
    tail = session_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _enforce_required_writes(session_id: str | None) -> None:
    """Fail a bug-verifier run that neither labels nor comments on the issue.

    The bug-verifier SOP guarantees at least a triage label on every verdict, and
    that write is what records a deferred operation for the finalize step to
    replay. An agent that only *describes* applying a label -- without ever
    invoking the tool -- produces a green run that never touches the issue and
    uploads no write-operations artifact. This turns that silent no-op into a hard
    failure.

    Sessions are resumed across triggers, so a re-run may legitimately write
    nothing because the label already landed in a prior run. In that case the
    issue already carries a triage label, which we accept.
    """
    if not session_id or not session_id.startswith("bug-verifier"):
        return

    invoked = get_invoked_write_tools()
    if _ISSUE_WRITE_TOOLS.intersection(invoked):
        # This run applied (or, in read-only mode, deferred) a label or comment.
        return

    # No issue-facing write this run. Accept only if a prior run's triage label
    # already landed on the issue (resumed, already-complete session).
    issue_number = _issue_number_from_session_id(session_id)
    if issue_number is not None:
        try:
            existing = get_issue_label_names(issue_number)
            if _TRIAGE_LABELS.intersection(existing):
                print(
                    f"ℹ️ No write this run, but issue #{issue_number} already carries a "
                    f"triage label {sorted(_TRIAGE_LABELS.intersection(existing))} from a "
                    "prior run -- accepting."
                )
                return
        except Exception as e:
            print(f"⚠️ Could not verify existing triage labels: {e}")

    raise RuntimeError(
        "bug-verifier finished without applying a label or comment to the issue, "
        "and no triage label from a prior run is present. The SOP mandates at least "
        f"a triage label on every verdict (write tools this run: {invoked or 'none'}). "
        "Nothing was applied to the issue and nothing was recorded for deferred "
        "execution -- failing so this is not a silent no-op."
    )


def _send_eval_trigger(session_id: str, eval_type: str) -> None:
    """Send evaluation trigger to SQS queue after agent completion.
    
    Only sends if EVALS_SQS_QUEUE_ARN environment variable is set.
    Derives queue URL from ARN (format: arn:aws:sqs:{region}:{account_id}:{queue_name}).
    
    Args:
        session_id: The unique session ID as stored in Langfuse (may include repo prefix).
        eval_type: The evaluation type (e.g., "reviewer", "implementer").
    """
    queue_arn = os.environ.get("EVALS_SQS_QUEUE_ARN")
    if not queue_arn:
        return
    
    # Parse ARN: arn:aws:sqs:{region}:{account_id}:{queue_name}
    arn_parts = queue_arn.split(":")
    if len(arn_parts) != 6:
        print(f"⚠️ Invalid SQS ARN format: {queue_arn}")
        return
    
    region = arn_parts[3]
    account_id = arn_parts[4]
    queue_name = arn_parts[5]
    queue_url = f"https://sqs.{region}.amazonaws.com/{account_id}/{queue_name}"
    
    try:
        sqs_client = boto3.client("sqs", region_name=region)
        message_body = json.dumps({
            "session_id": session_id,
            "eval_type": eval_type
        })
        sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=message_body
        )
        print(f"✅ Sent eval trigger to SQS: {message_body}")
    except Exception as e:
        print(f"⚠️ Failed to send eval trigger to SQS: {e}")


def _setup_langfuse_telemetry() -> bool:
    """Set up Langfuse telemetry if environment variables are configured.
    
    Returns:
        True if telemetry was successfully configured, False otherwise.
    """
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    langfuse_host = os.environ.get("LANGFUSE_HOST")
    
    if not all([langfuse_public_key, langfuse_secret_key, langfuse_host]):
        print("ℹ️ Langfuse telemetry not configured (missing environment variables)")
        return False
    
    try:
        langfuse_auth = base64.b64encode(
            f"{langfuse_public_key}:{langfuse_secret_key}".encode()
        ).decode()
        
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{langfuse_host}/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
        
        StrandsTelemetry().setup_otlp_exporter()
        print("✅ Langfuse telemetry configured successfully")
        return True
    except Exception as e:
        print(f"⚠️ Failed to configure Langfuse telemetry: {e}")
        return False


def _get_trace_attributes() -> dict:
    """Build trace attributes from environment context."""
    session_id = os.getenv("SESSION_ID", "")
    github_actor = os.getenv("GITHUB_ACTOR", "")
    github_repository = os.getenv("GITHUB_REPOSITORY", "")
    github_workflow = os.getenv("GITHUB_WORKFLOW", "")
    github_run_id = os.getenv("GITHUB_RUN_ID", "")
    
    # Include repo name in session ID for uniqueness across repos
    # Format: "owner_repo:session-id" (e.g., "strands-agents_sdk-typescript:reviewer-443")
    repo_prefix = github_repository.replace("/", "_") if github_repository else "unknown"
    unique_session_id = f"{repo_prefix}:{session_id}" if session_id else f"{repo_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "session.id": unique_session_id,
        "user.id": github_actor,
        "langfuse.tags": [
            f"repo:{github_repository}",
            f"workflow:{github_workflow}",
            f"run:{github_run_id}",
            "strands-github-agent",
        ],
    }

def _get_all_tools() -> list[Any]:
    return [
        # File editing
        str_replace_based_edit_tool,
        
        # System tools
        shell,
        http_request,
        
        # GitHub issue tools
        create_issue,
        get_issue,
        update_issue,
        list_issues,
        add_issue_comment,
        add_issue_labels,
        get_issue_comments,
        
        # GitHub PR tools
        create_pull_request,
        get_pull_request,
        update_pull_request,
        list_pull_requests,
        get_pr_files,
        get_pr_review_and_comments,
        reply_to_review_comment,
        add_pr_comment,
        
        # Agent tools
        notebook,
        handoff_to_user,
    ]


def run_agent(query: str):
    """Run the agent with the provided query."""
    try:
        # Set up Langfuse telemetry (optional - gracefully degrades if not configured)
        telemetry_enabled = _setup_langfuse_telemetry()
        trace_attributes = _get_trace_attributes() if telemetry_enabled else {}
        
        # Get tools and create model
        tools = _get_all_tools()
        
        # Create Bedrock model with inlined configuration
        additional_request_fields = {}
        additional_request_fields["thinking"] = {"type": "adaptive"}
        additional_request_fields["output_config"] = {"effort": "high"}
        
        model = BedrockModel(
            model_id=STRANDS_MODEL_ID,
            max_tokens=STRANDS_MAX_TOKENS,
            region_name=STRANDS_REGION,
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
            cache_config=CacheConfig(strategy="auto"),
            additional_request_fields=additional_request_fields,
            cache_prompt="default",
            cache_tools="default",
        )
        system_prompt = os.getenv("INPUT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        session_id = os.getenv("SESSION_ID")
        s3_bucket = os.getenv("S3_SESSION_BUCKET")
        s3_prefix = os.getenv("GITHUB_REPOSITORY", "")

        if s3_bucket and session_id:
            print(f"🤖 Using session manager with session ID: {session_id}")
            session_manager = S3SessionManager(
                session_id=session_id,
                bucket=s3_bucket,
                prefix=s3_prefix,
            )
        else:
            raise ValueError("Both SESSION_ID and S3_SESSION_BUCKET must be set")

        # Offload oversized tool results (large PR diffs, file reads, shell output)
        # to storage instead of letting them crowd out the context window. The
        # plugin registers a retrieval tool so the agent can fetch the full content
        # on demand. Use S3 storage rather than in-memory: the offloaded result is
        # replaced in the conversation with a storage reference that the session
        # manager persists, so a resumed session must still be able to resolve that
        # reference in a later process — which in-memory storage could not.
        context_offloader = ContextOffloader(
            storage=S3Storage(bucket=s3_bucket, prefix=f"{s3_prefix}/offload"),
        )

        # Create agent with optional trace attributes for Langfuse
        agent_kwargs = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": tools,
            "session_manager": session_manager,
            "plugins": [context_offloader],
        }

        if trace_attributes:
            agent_kwargs["trace_attributes"] = trace_attributes

        agent = Agent(**agent_kwargs)

        print("Processing user query...")
        result = agent(query)

        print(f"\n\nAgent Result 🤖\nStop Reason: {result.stop_reason}\nMessage: {json.dumps(result.message, indent=2)}")

        # Fail loudly if a mode with a mandatory write (e.g. bug-verifier) finished
        # without actually invoking it, rather than ending as a green no-op.
        _enforce_required_writes(session_id)

        # Use the unique session ID from trace attributes (includes repo prefix)
        unique_session_id = trace_attributes.get("session.id", session_id)
        eval_type = session_id.split("-")[0] if "-" in session_id else session_id
        _send_eval_trigger(unique_session_id, eval_type)
    except Exception as e:
        error_msg = f"❌ Agent execution failed: {e}"
        print(error_msg)
        raise e


def main() -> None:
    """Main entry point for the agent runner."""
    try:
        # Read task from command line arguments
        if len(sys.argv) < 2:
            raise ValueError("Task argument is required")

        task = " ".join(sys.argv[1:])
        if not task.strip():
            raise ValueError("Task cannot be empty")
        print(f"🤖 Running agent with task: {task}")

        run_agent(task)

    except Exception as e:
        error_msg = f"Fatal error: {e}"
        print(error_msg)

        sys.exit(1)


if __name__ == "__main__":
    main()
