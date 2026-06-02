# Connecting your GitHub account

Colony agents commit code on your behalf and assign GitHub issues
back to you. For either of those to "stick" — for the commits to
show your avatar in GitHub's UI, for the issues to land on your
real GitHub profile — Colony needs to know who *you* are on GitHub.

Click **Connect GitHub** on your profile, approve at GitHub, you're
done. The rest of this page explains exactly what that button does,
what Colony stores, and what doesn't work until you click it.

> **Prereqs before this button works:**
>
> - **Service provider** (the team running this Colony deployment)
>   has registered the Colony GitHub App and set
>   `GITHUB_APP_CLIENT_ID` + `GITHUB_APP_CLIENT_SECRET` in the
>   deployment's `.env` —
>   [`github-app-setup.md`](github-app-setup.md) §1–§2.
> - **Tenant admin** (a GitHub-org admin in your enterprise) has
>   installed the same App into your tenant's GitHub org and saved
>   the installation id in the dashboard —
>   [`github-app-setup.md`](github-app-setup.md) §3.
>
> If either is missing, the **Connect GitHub** button either 503s
> or the OAuth round-trip succeeds but subsequent Colony actions
> against your tenant's repos still fail with auth errors. Ask
> your tenant admin first; they can tell whether their step is
> done.

## What happens when you click "Connect GitHub"

1. The button (in the **Your GitHub Identity** panel on the
   landing page) redirects your browser to GitHub.
2. GitHub shows you what the Colony App is asking for. For this
   flow it asks to *Read your email addresses* — that's it. No
   write access, no organisation access, nothing risky.
3. You approve.
4. GitHub redirects you back to Colony's callback URL with a
   one-shot authorisation code.
5. Colony's backend exchanges the code for a short-lived
   user-to-server token, calls `GET /user` + `GET /user/emails`
   on your behalf, and reads:
   - your **GitHub login** (e.g. `anassar`),
   - your **GitHub user id** (numeric, stable across renames),
   - your **verified primary email** (picked from your verified
     addresses on GitHub),
   - your **display name**.
6. Those four values get persisted on the `users` row in
   Postgres. **The user-to-server token is discarded** — Colony
   never acts AS you on GitHub.
7. You're returned to a confirmation page showing the verified
   identity.

The whole round-trip takes under a second on the happy path.

## Why typing your GitHub login/email isn't an option

The earlier version of Colony had a settings UI where you typed
your name + email. We removed it because it enabled commit
impersonation: GitHub's UI matches commit emails to user accounts
to display avatars, so typing `linus@kernel.org` into the Colony
profile would produce commits that GitHub would attribute to
Linus Torvalds. Verifying via OAuth — where GitHub itself tells
us which login + verified emails belong to you — is the only
honest path.

## What Colony does once you're connected

| Surface | How your identity is used |
|---|---|
| **Commit attribution** (per colony preference; see [`git-attribution.md`](git-attribution.md)) | When `commit_principal=user` or `commit_co_author=user`, your `git_user_name` + `github_email` go into the `Author:` line or the `Co-authored-by:` trailer. GitHub matches the email to your account → your avatar shows on the commit. |
| **Issue assignment** ([`propose_task_assignments`](../../src/polymathera/colony/design_monorepo/process.py)) | When the planner classifies a task as user-owned (either via the explicit `<!-- colony:assignee: user -->` marker on a roadmap line or via LLM classification), the issue gets assigned to your `github_login` on GitHub. You receive the standard GitHub notifications. |
| **Issue comments** ([`comment_as_session_agent`](../../src/polymathera/colony/agents/patterns/capabilities/github.py)) | When Colony replies to an issue you raised, the comment includes a footer naming you (your `github_login`) as the human the bot is replying to. |

The Colony bot identity (`<app-slug>[bot]`) is what GitHub sees
as the actor — Colony doesn't post AS you. Your identity surfaces
in the commit Co-Authored-By trailer + the issue assignee field
+ the comment footers.

## What doesn't work until you connect

- **Commits show only the colony's synthetic identity.** When
  `commit_principal=user` is set on a colony and you haven't
  OAuth'd, `_resolve_attribution` falls through with the
  user-side dropped; the commit succeeds with the
  `colony:<colony_id>` synthetic identity as `Author:`, no
  trailer. (See [`git-attribution.md` failure modes](git-attribution.md#failure-modes).)
- **`propose_task_assignments` marks your tasks `user_unassignable=True`.**
  The action's apply step skips them — no `assign_issue` call
  is made, and they don't appear in `applied` or `errors`.
  Stats include a `user_unassignable_count` so the mission's
  caller can surface the gap.
- **Issue comment footers don't include your login.** They fall
  back to anonymous wording.

None of these block agents from working. They just mean
contributions land without your name on them.

## Re-verify or disconnect

The same **Your GitHub Identity** panel exposes two buttons once
you're connected:

- **Re-verify** — runs the OAuth flow again. Use this after
  changing your primary email on GitHub, after rotating into a
  new account, or whenever GitHub asks you to re-approve.
- **Disconnect** — clears every GitHub-side field on your `users`
  row (`github_login`, `github_user_id`, `github_email`,
  `git_user_name`, `github_connected_at`, `github_last_verified_at`).
  Subsequent sessions see no identity until you re-connect.
  Idempotent — disconnecting an already-disconnected user is a
  no-op.

Disconnecting does not retroactively scrub commits or issues
that were already attributed under your name; it only stops
future ones.

## Privacy

- Colony stores only the four fields listed in step 6 (login,
  user id, primary email, display name). No repo list, no
  history, no scopes beyond `user:email`.
- The OAuth token is held in memory for the round-trip and
  discarded — it's not written to disk or persisted in any DB.
- The Tenant GitHub Installation is **separate** from your
  per-user connection. The installation grants Colony access to
  the tenant's repos and is set by the tenant admin; your
  connection is only about your identity being recognized on
  attributions and assignments.
- Disconnecting clears your row but does not uninstall the App
  from your tenant's org. Reach out to your tenant admin if you
  want Colony's access to the org revoked.

## Related

- [`github-app-setup.md`](github-app-setup.md) — operator-side
  setup that this user flow depends on.
- [`git-attribution.md`](git-attribution.md) — how the
  per-colony commit_principal/co_author preference combines with
  your connected identity.
