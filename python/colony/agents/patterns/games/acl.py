"""Agent Communication Language (ACL) with speech act semantics.

Based on FIPA ACL and speech act theory (Austin, Searle), as discussed in
Wooldridge and `MULTI_AGENT_GAME_ENGINE.md`.

Messages have:
- Performative (illocutionary force): inform, request, propose, challenge, etc.
- Content: Typed payload with schema
- Preconditions: What must be true for message to be valid
- Expected effects: What should happen after message

This enables:
- Structured communication with clear semantics
- Validation of message preconditions
- Detection of miscommunication
- Enforcement of protocol rules
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Performative(str, Enum):
    """Speech act performatives (illocutionary force).

    Based on FIPA ACL and speech act theory.
    """

    # Assertive (stating facts)
    INFORM = "inform"  # Inform about a fact
    CONFIRM = "confirm"  # Confirm something is true
    DISCONFIRM = "disconfirm"  # State something is false

    # Directive (requesting action)
    REQUEST = "request"  # Request an action
    QUERY = "query"  # Request information
    COMMAND = "command"  # Command an action (strong request)

    # Commissive (committing to action)
    PROPOSE = "propose"  # Propose a solution/hypothesis
    COMMIT = "commit"  # Commit to doing something
    PROMISE = "promise"  # Promise to deliver something
    OFFER = "offer"  # Offer to do something

    # Permissive (granting/denying)
    ACCEPT = "accept"  # Accept a proposal/offer
    REJECT = "reject"  # Reject a proposal/offer
    AGREE = "agree"  # Agree with a statement
    DISAGREE = "disagree"  # Disagree with a statement

    # Challenging (skeptical)
    CHALLENGE = "challenge"  # Challenge a claim/hypothesis
    QUESTION = "question"  # Question assumptions
    REFUTE = "refute"  # Refute with counter-evidence

    # Meta-communication
    RETRACT = "retract"  # Retract previous message
    CLARIFY = "clarify"  # Clarify previous message
    ACKNOWLEDGE = "acknowledge"  # Acknowledge receipt

    # Answering
    ANSWER = "answer"  # Answer to a query
    EXPLAIN = "explain"  # Provide explanation


class MessageSchema(str, Enum):
    """Content schema types for message payloads.

    Defines ontology for agent communication.
    """

    # Hypothesis-related
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    CHALLENGE_CLAIM = "challenge_claim"

    # Task-related
    TASK_SPEC = "task_spec"
    TASK_BID = "task_bid"
    TASK_RESULT = "task_result"

    # Analysis-related
    FINDING = "finding"
    ANALYSIS_RESULT = "analysis_result"
    SCOPE_AWARE_RESULT = "scope_aware_result"

    # Validation-related
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESULT = "validation_result"

    # Coordination-related
    PLAN = "plan"
    ACTION = "action"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"

    # Voting-related
    VOTE = "vote"
    RANKING = "ranking"

    # Negotiation-related
    OFFER_PROPOSAL = "offer_proposal"
    COUNTER_OFFER = "counter_offer"

    # Generic
    STRUCTURED_DATA = "structured_data"
    FREE_TEXT = "free_text"


class ACLMessage(BaseModel):
    """Agent Communication Language message.

    Following FIPA ACL design with speech act semantics.

    Examples:
        Inform about finding:
        ```python
        msg = ACLMessage(
            performative=Performative.INFORM,
            sender="analyzer_001",
            receivers=["coordinator_001"],
            content={
                "schema": MessageSchema.FINDING,
                "payload": {
                    "finding_type": "security_issue",
                    "description": "SQL injection vulnerability",
                    "location": "auth.py:42"
                }
            },
            preconditions=["analysis_completed"],
            expected_effect="update_findings_list"
        )
        ```

        Challenge hypothesis:
        ```python
        msg = ACLMessage(
            performative=Performative.CHALLENGE,
            sender="skeptic_002",
            receivers=["proposer_001"],
            conversation_id="hyp_game_123",
            in_reply_to="msg_propose_hyp",
            content={
                "schema": MessageSchema.CHALLENGE_CLAIM,
                "payload": {
                    "challenged_claim": "Function never returns null",
                    "reason": "Missing error handling for edge case",
                    "counter_evidence": ["Line 55: null returned on timeout"]
                }
            },
            preconditions=["hypothesis_proposed"],
            expected_effect="defend_hypothesis_or_revise"
        )
        ```
    """

    # Identification
    message_id: str = Field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex}",
        description="Unique message identifier"
    )

    # Speech act
    performative: Performative = Field(
        description="Illocutionary force of the message"
    )

    # Participants
    sender: str = Field(
        description="Sending agent ID"
    )

    receivers: list[str] | Literal["all"] = Field(
        description="Receiving agent IDs or 'all' for broadcast"
    )

    # Conversation tracking
    conversation_id: str = Field(
        default_factory=lambda: f"conv_{uuid.uuid4().hex}",
        description="Conversation/game ID this message belongs to"
    )

    in_reply_to: str | None = Field(
        default=None,
        description="Message ID this is replying to"
    )

    # Content (typed payload)
    content: dict[str, Any] = Field(
        description="Message content with 'schema' and 'payload' fields"
    )

    # Speech act semantics
    preconditions: list[str] = Field(
        default_factory=list,
        description="Conditions that should hold for message to be valid"
    )

    expected_effect: str | None = Field(
        default=None,
        description="Expected effect of this message (belief update, goal creation, etc.)"
    )

    # Metadata
    priority: int = Field(
        default=0,
        description="Message priority (higher = more urgent)"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When message was created"
    )

    expires_at: float | None = Field(
        default=None,
        description="When message expires (None = no expiration)"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata"
    )

    # Validation
    validated: bool = Field(
        default=False,
        description="Whether message has been validated"
    )

    def is_expired(self) -> bool:
        """Check if message is expired.

        Returns:
            True if expired
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_schema(self) -> str | None:
        """Get content schema.

        Returns:
            Schema name or None
        """
        return self.content.get("schema")

    def get_payload(self) -> dict[str, Any]:
        """Get content payload.

        Returns:
            Payload dictionary
        """
        return self.content.get("payload", {})

    def is_reply_to(self, message_id: str) -> bool:
        """Check if this is a reply to given message.

        Args:
            message_id: Message ID to check

        Returns:
            True if this replies to that message
        """
        return self.in_reply_to == message_id

    def validate_preconditions(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate message preconditions against context.

        Args:
            context: Current context (game state, beliefs, etc.)

        Returns:
            (all_satisfied, unsatisfied_conditions) tuple
        """
        unsatisfied = []

        for precondition in self.preconditions:
            # Check if precondition is satisfied
            # Simplified check - real implementation would be more sophisticated
            if precondition not in context or not context[precondition]:
                unsatisfied.append(precondition)

        return (len(unsatisfied) == 0, unsatisfied)


# Utility functions

def create_inform_message(
    sender: str,
    receivers: list[str] | Literal["all"],
    content_schema: MessageSchema,
    payload: dict[str, Any],
    conversation_id: str | None = None
) -> ACLMessage:
    """Create an INFORM message.

    Args:
        sender: Sending agent
        receivers: Receiving agents
        content_schema: Content schema
        payload: Content payload
        conversation_id: Optional conversation ID

    Returns:
        ACL message
    """
    return ACLMessage(
        performative=Performative.INFORM,
        sender=sender,
        receivers=receivers,
        conversation_id=conversation_id or f"conv_{uuid.uuid4().hex}",
        content={
            "schema": content_schema.value,
            "payload": payload
        },
        preconditions=["has_evidence"],  # INFORM should have evidence
        expected_effect="update_belief"
    )


def create_request_message(
    sender: str,
    receivers: list[str],
    content_schema: MessageSchema,
    payload: dict[str, Any],
    conversation_id: str | None = None
) -> ACLMessage:
    """Create a REQUEST message.

    Args:
        sender: Sending agent
        receivers: Receiving agents
        content_schema: Content schema
        payload: Content payload
        conversation_id: Optional conversation ID

    Returns:
        ACL message
    """
    return ACLMessage(
        performative=Performative.REQUEST,
        sender=sender,
        receivers=receivers,
        conversation_id=conversation_id or f"conv_{uuid.uuid4().hex}",
        content={
            "schema": content_schema.value,
            "payload": payload
        },
        expected_effect="create_goal"
    )


def create_propose_message(
    sender: str,
    receivers: list[str] | Literal["all"],
    hypothesis: dict[str, Any],
    conversation_id: str
) -> ACLMessage:
    """Create a PROPOSE message (for hypothesis game).

    Args:
        sender: Proposing agent
        receivers: Receiving agents
        hypothesis: Hypothesis content
        conversation_id: Conversation ID

    Returns:
        ACL message
    """
    return ACLMessage(
        performative=Performative.PROPOSE,
        sender=sender,
        receivers=receivers,
        conversation_id=conversation_id,
        content={
            "schema": MessageSchema.HYPOTHESIS.value,
            "payload": hypothesis
        },
        preconditions=["hypothesis_well_formed", "has_evidence"],
        expected_effect="initiate_validation"
    )


def create_challenge_message(
    sender: str,
    receivers: list[str],
    challenged_claim: str,
    reason: str,
    counter_evidence: list[str],
    conversation_id: str,
    in_reply_to: str
) -> ACLMessage:
    """Create a CHALLENGE message (for hypothesis game).

    Args:
        sender: Challenging agent
        receivers: Receiving agents
        challenged_claim: What is being challenged
        reason: Why it's being challenged
        counter_evidence: Evidence against claim
        conversation_id: Conversation ID
        in_reply_to: Message being challenged

    Returns:
        ACL message
    """
    return ACLMessage(
        performative=Performative.CHALLENGE,
        sender=sender,
        receivers=receivers,
        conversation_id=conversation_id,
        in_reply_to=in_reply_to,
        content={
            "schema": MessageSchema.CHALLENGE_CLAIM.value,
            "payload": {
                "challenged_claim": challenged_claim,
                "reason": reason,
                "counter_evidence": counter_evidence
            }
        },
        preconditions=["hypothesis_proposed"],
        expected_effect="defend_or_revise"
    )


def create_accept_message(
    sender: str,
    receivers: list[str],
    accepted_message_id: str,
    conversation_id: str,
    reasoning: str | None = None
) -> ACLMessage:
    """Create an ACCEPT message.

    Args:
        sender: Accepting agent
        receivers: Receiving agents
        accepted_message_id: Message being accepted
        conversation_id: Conversation ID
        reasoning: Optional reasoning for acceptance

    Returns:
        ACL message
    """
    return ACLMessage(
        performative=Performative.ACCEPT,
        sender=sender,
        receivers=receivers,
        conversation_id=conversation_id,
        in_reply_to=accepted_message_id,
        content={
            "schema": MessageSchema.STRUCTURED_DATA.value,
            "payload": {
                "accepted_message_id": accepted_message_id,
                "reasoning": reasoning
            }
        },
        expected_effect="finalize_agreement"
    )


def create_reject_message(
    sender: str,
    receivers: list[str],
    rejected_message_id: str,
    reason: str,
    conversation_id: str
) -> ACLMessage:
    """Create a REJECT message.

    Args:
        sender: Rejecting agent
        receivers: Receiving agents
        rejected_message_id: Message being rejected
        reason: Reason for rejection
        conversation_id: Conversation ID

    Returns:
        ACL message
    """
    return ACLMessage(
        performative=Performative.REJECT,
        sender=sender,
        receivers=receivers,
        conversation_id=conversation_id,
        in_reply_to=rejected_message_id,
        content={
            "schema": MessageSchema.STRUCTURED_DATA.value,
            "payload": {
                "rejected_message_id": rejected_message_id,
                "reason": reason
            }
        },
        expected_effect="revise_or_abandon"
    )


# Integration with existing communication infrastructure

def acl_to_message(
    acl_msg: ACLMessage,
    message_type: str | None = None,
    channel: str = "direct"
) -> dict[str, Any]:
    """Convert ACL message to infrastructure Message format.

    ACL messages are high-level semantic messages that should be wrapped
    in the infrastructure-level Message for actual delivery through the
    communication system.

    Args:
        acl_msg: ACL message to convert
        message_type: Optional message type override (default: inferred from performative)
        channel: Communication channel (default: "direct")

    Returns:
        Dictionary ready for Message model instantiation

    Example:
        from polymathera.colony.agents.models import Message

        acl_msg = create_inform_message(...)
        msg_dict = acl_to_message(acl_msg)
        msg = Message(**msg_dict)

        # Or directly:
        msg = Message(
            **acl_to_message(acl_msg),
            from_agent_id=agent.id  # Can override fields
        )
    """
    # Infer message type from performative if not provided
    if message_type is None:
        message_type = _infer_message_type(acl_msg.performative)

    return {
        "message_id": acl_msg.message_id,
        "from_agent_id": acl_msg.sender,
        "to_agent_id": acl_msg.receivers[0] if len(acl_msg.receivers) == 1 else acl_msg.receivers,
        "message_type": message_type,
        "channel": channel,
        "content": acl_msg.model_dump(),  # ACL message as content
        "requires_response": acl_msg.performative in {
            Performative.REQUEST,
            Performative.QUERY,
            Performative.PROPOSE,
            Performative.CFP
        },
        "response_to": acl_msg.in_reply_to,
        "metadata": {
            "conversation_id": acl_msg.conversation_id,
            "performative": acl_msg.performative.value,
            "protocol": acl_msg.protocol
        }
    }


def message_to_acl(msg_data: dict[str, Any]) -> ACLMessage:
    """Extract ACL message from infrastructure Message content.

    Args:
        msg_data: Message content (should contain ACL message dict)

    Returns:
        ACL message

    Example:
        # When receiving a message:
        msg = await mailbox.receive()
        acl_msg = message_to_acl(msg.content)

        if acl_msg.performative == Performative.PROPOSE:
            handle_proposal(acl_msg)
    """
    return ACLMessage(**msg_data)


def _infer_message_type(performative: Performative) -> str:
    """Infer infrastructure message type from ACL performative.

    Args:
        performative: ACL performative

    Returns:
        Message type string
    """
    # Map performatives to infrastructure message types
    if performative in {Performative.REQUEST, Performative.QUERY, Performative.CFP}:
        return "request"
    elif performative in {Performative.INFORM, Performative.CONFIRM, Performative.DISCONFIRM}:
        return "notification"
    elif performative in {Performative.PROPOSE, Performative.OFFER, Performative.COMMIT}:
        return "response"
    elif performative == Performative.COMMAND:
        return "command"
    else:
        return "notification"  # Default


