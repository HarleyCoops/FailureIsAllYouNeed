# W&B evidence and MCP connection

The two experiments remain separate W&B projects:

- [Qwen tool-assisted repair](https://wandb.ai/christian-cooper-us/weight-space-breakout/runs/7a31ec5872dd3f77)
- [Kimi analytic parameter repair](https://forge.coreweave.com/wandb/christian-cooper-us/kimi-brain-surgery)

The official [W&B MCP server](https://github.com/wandb/wandb-mcp-server) is
configured in the local Codex client using its remote endpoint and an existing
environment variable:

```bash
codex mcp add wandb --url https://mcp.withwandb.com/mcp --bearer-token-env-var WANDB_API_KEY
```

The API key itself is not committed to this repository or stored in that
configuration entry. The variable must be available to the Codex process.
A new chat or app restart may be needed to expose newly configured tools.

The connection was tested with real MCP initialization and queries:

- The Qwen run's return metrics, acceptance/deployment status, and selected
  policy hash matched the published evidence. See the
  [verification receipt](../results/qualified-repair-01/wandb-verification.json).
- Kimi's 20 finished runs were retrieved. All 18 sweep summaries matched the
  pinned source files; v1 and v4 paired summaries matched the per-seed records.
  See [provenance](../results/kimi/provenance.json).
- Returned points from the monotonic run's history matched the saved
  candidate scores and keep/rollback flags. The response's sampling metadata
  is retained in the [history receipt](../results/kimi/wandb-monotonic-history.json).

This update reads existing runs and publishes reproducible figures. It does
not create duplicate training runs or change any model weights. Future
combined experiments should log candidate and incumbent metrics separately,
record acceptance decisions and checkpoint hashes, and identify each metric's
practice/gate/audit split.

## Reconciled reports — 2026-11-03

[Qwen modes and audit](https://wandb.ai/christian-cooper-us/weight-space-breakout/reports/Evidence-reconciliation---2026-11-03--VmlldzoxODI1MTI2Mw==) · [Kimi outcomes and batch sweep](https://wandb.ai/christian-cooper-us/kimi-brain-surgery/reports/Evidence-reconciliation---2026-11-03--VmlldzoxODI1MTI2NA==).

Original metrics are preserved. Posthoc greedy metrics are explicitly namespaced.
The batch sweep has 18 finished runs, all checked against its source JSON.
