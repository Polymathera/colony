"""``POST /api/v1/github/webhook`` — receiver for GitHub App webhook
deliveries.

Per design doc §10:

1. Verify ``X-Hub-Signature-256`` HMAC against ``GITHUB_WEBHOOK_SECRET``
   (constant-time compare).
2. INSERT ``X-GitHub-Delivery`` into ``github_webhook_deliveries``
   with ``ON CONFLICT DO NOTHING`` — retries are a no-op.
3. Look up the tenant by ``installation.id`` in the payload.
4. Normalize the payload into ``(GitHubEventProtocol key, value)``.
5. Fan-out the write to every colony in that tenant.

Returns 200 with a small status JSON in every "the request was
well-formed" case (accepted / duplicate / ignored / no-tenant). Only
401 (bad HMAC) and 503 (receiver disabled — env var unset) are
non-200. GitHub retries only on non-2xx, so the 200-with-status
shape prevents redundant retries for cases we've already handled.

NOT mounted unless ``GITHUB_WEBHOOK_SECRET`` is set in env — the
route's first check is "is the receiver configured", and an unset
secret short-circuits to 503 with a clear setup-doc pointer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection
from ..auth import service as auth_service
from ..github_webhook.normalizer import HANDLED_EVENT_TYPES, normalize
from ..github_webhook.publisher import publish_to_tenant_colonies
from ..github_webhook.schema import record_delivery


logger = logging.getLogger(__name__)
router = APIRouter()


def _verify_hmac(raw_body: bytes, header_value: str, secret: str) -> bool:
    """Constant-time HMAC-SHA256 verification of ``X-Hub-Signature-256``.

    GitHub formats the header as ``sha256=<hex>``. We compute the
    same digest over the raw request body and compare via
    :func:`hmac.compare_digest`. Returns ``False`` for any malformed
    input — never raises on operator-controlled inputs.
    """

    if not secret or not header_value or not header_value.startswith("sha256="):
        return False
    expected_hex = header_value.removeprefix("sha256=")
    digest = hmac.new(
        secret.encode("utf-8"), raw_body, hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected_hex, digest)


@router.post("/github/webhook")
async def github_webhook(
    request: Request,
    colony: ColonyConnection = Depends(get_colony),
    x_github_event: str = Header(..., alias="X-GitHub-Event"),
    x_github_delivery: str = Header(..., alias="X-GitHub-Delivery"),
    x_hub_signature_256: str = Header(..., alias="X-Hub-Signature-256"),
) -> dict[str, str]:
    """GitHub App webhook receiver. See module docstring for the
    full pipeline."""

    # Step 0: receiver enabled?
    from polymathera.colony.agents.configs import get_github_auth_config
    gh_auth = await get_github_auth_config()
    if not gh_auth.webhook_secret:
        raise HTTPException(
            status_code=503,
            detail=(
                "GitHub webhook receiver is disabled — operator has "
                "not set ``GITHUB_WEBHOOK_SECRET`` on the dashboard "
                "service. See docs/guides/github-app-setup.md."
            ),
        )

    raw_body = await request.body()

    # Step 1: HMAC verify.
    if not _verify_hmac(raw_body, x_hub_signature_256, gh_auth.webhook_secret):
        logger.warning(
            "github_webhook: HMAC verify failed for delivery=%s "
            "event=%s (length=%d)",
            x_github_delivery, x_github_event, len(raw_body),
        )
        raise HTTPException(status_code=401, detail="invalid signature")

    db_pool = colony._db_pool
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available — dashboard misconfigured.",
        )

    # Step 2: dedup.
    inserted = await record_delivery(
        db_pool, delivery_id=x_github_delivery, event_type=x_github_event,
    )
    if not inserted:
        # GitHub retry of a delivery we already processed.
        return {"status": "duplicate"}

    # Step 3+4: parse + normalize. Body parse failures are surfaced
    # as 400 since they indicate a malformed payload from GitHub
    # (very rare; usually means the operator pointed the receiver
    # at a non-GitHub HTTP client).
    try:
        payload = json.loads(raw_body)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"payload is not JSON: {exc}",
        ) from exc

    if x_github_event not in HANDLED_EVENT_TYPES:
        # Known-unhandled event type (ping, discussion, etc.). 200 so
        # GitHub doesn't retry; status surfaces the no-op.
        return {"status": "ignored", "event": x_github_event}

    normalized = normalize(x_github_event, payload)
    if normalized is None:
        logger.warning(
            "github_webhook: normalizer produced no event for "
            "event=%s delivery=%s (payload malformed?)",
            x_github_event, x_github_delivery,
        )
        return {"status": "malformed_payload"}
    key, value = normalized

    # Step 5: tenant lookup → colony fan-out.
    installation = payload.get("installation") or {}
    installation_id = installation.get("id")
    if installation_id is None:
        return {"status": "no_installation"}

    tenant_row = await auth_service.get_tenant_by_installation_id(
        db_pool, installation_id=installation_id,
    )
    if tenant_row is None:
        # We've installed for this installation_id — but no tenant
        # row points at it. Likely a stale install (uninstalled,
        # re-installed elsewhere) or operator hasn't set the panel.
        logger.warning(
            "github_webhook: no tenant configured for "
            "installation_id=%s (delivery=%s)",
            installation_id, x_github_delivery,
        )
        return {"status": "no_tenant_for_installation"}

    tenant_id = tenant_row["tenant_id"]
    colonies = await auth_service.list_colonies(db_pool, tenant_id)
    colony_ids = [c["colony_id"] for c in colonies]
    if not colony_ids:
        return {"status": "no_colonies_in_tenant"}

    written = await publish_to_tenant_colonies(
        app_name=colony.app_name,
        tenant_id=tenant_id,
        colony_ids=colony_ids,
        key=key,
        value=value,
        delivery_id=x_github_delivery,
    )
    logger.info(
        "github_webhook: delivered event=%s key=%s to %d/%d colonies "
        "(tenant=%s, delivery=%s)",
        x_github_event, key, written, len(colony_ids),
        tenant_id, x_github_delivery,
    )
    return {"status": "accepted", "colonies_written": str(written)}
