# MiniMax-M3.1 PR CI (nho2)

This is the `Inferact/vllm-internal` `minimax-m31-prod` CI (pipeline
`vllm-internal-minimax-m31-ci`), ported to `nho_tracking`. It runs on the same
agents and the same evals.

Comment exactly `/ci run` on a pull request that targets `nho_tracking` to
start the `nho2-minimax-m31-ci` Buildkite pipeline on the PR head. The
`.github/workflows/minimax-m31-ci.yml` workflow on `main` handles the comment:
it reacts with :eyes:, creates the build through the Buildkite API (secret
`BUILDKITE_MINIMAX_M31_CI_TOKEN`), and replies with the build link. Buildkite's
own comment trigger takes a single word only, so it cannot match `/ci run`.
New commits do not start builds; comment `/ci run` again to test the new head.
PRs from forks are rejected.

The bootstrap (`bootstrap.yaml`, the pipeline's live configuration) and the
orchestration in this directory are loaded from the latest `nho_tracking`,
so only the PR's source commit is tested, never its CI scripts. API builds may
set `MINIMAX_M31_CI_REF` to load the orchestration from another branch, which
is how changes to this directory are validated before merge.

## What a build does

1. **Image.** Triggers the `image-build` pipeline (`Inferact/image-build`,
   profile `minimax-m31-x86`). For an `Inferact/nho2` source it builds the PR
   head on a `large_cpu_x86_64` agent with the `docker_build_cmd` of
   `builds/minimax-m31-nvda/nho2-minimax-m31.yaml` (CUDA 13.4, torch nightly,
   MSA from the public `vllm-project/MSA` pin in `fmha_sm100.cmake`) and pushes
   `inferactinc/dev:nho2-pr-<pr>-<sha12>-x86-cu134-minimax-m31`.
2. **Evals.** One job on the `b300-tp4` queue runs `m31-tp4-ci.yaml` with vigil
   in local mode. vLLM serves `Inferact/Minimax-M3-NVFP4` from the PR image
   with the prod config from `examples/minimax-m31/nvidia/m31-tp4_dspark_spec3.yaml`
   (TP4, NVFP4 KV + NVFP4 indexer, DSpark spec3). mqa then runs
   GPQA Diamond (4 epochs), AIME25 (16 epochs) and MMMU Pro. Scores go to the
   build annotation. mqa results, interactions and server logs are uploaded as
   artifacts. The job fails if vigil fails or a score in `thresholds.yaml`
   is below its minimum.

vigil and mqa are pinned in `package-deps.sh`. The CPU generator step exports
them into the `minimax-m31-ci.tar.gz` artifact, so the GPU agents need no GitHub
credentials.

## Agents

`b300-tp4` has four agents, two on each of `verda-b300-01` and
`verda-b300-02` (drained from Slurm). Each agent owns four GPUs and one server
port. Its systemd unit (`/etc/systemd/system/buildkite-agent-{0,1}.service`)
sets:

| Variable | Agent 0 | Agent 1 |
| --- | --- | --- |
| `CI_GPU_DEVICES` | `0,1,2,3` | `4,5,6,7` |
| `CI_SERVE_PORT` | `8000` | `8100` |
| `CI_HF_HOME` | `/mnt/shared/hf-models` | `/mnt/shared/hf-models` |

`agents/setup-agents.sh` installs them. `/etc/buildkite-agent/hooks/environment`
exports `HF_TOKEN` for the gated eval datasets (GPQA needs an account that
accepted its terms). Build and temp directories are under `/home/buildkite-agent`. Docker
images stay on local disk: the root disk on `verda-b300-02`, and
`/mnt/nvme/docker` on `verda-b300-01` (`data-root` with the containerd image
store off, as its root disk is full). Each job keeps 150 GB free by dropping
older PR images of both MiniMax-M3.1 CIs. The model weights are pre-downloaded to `/mnt/shared/hf-models`
and served offline.
Results are pushed to the MQA dashboard (<https://mqa.inferact.ai>) as
`buildkite/nho2-minimax-m31-ci`, with the PR, commit and build URL in
the note. The Cloudflare Access service token (`CF_ACCESS_CLIENT_ID`,
`CF_ACCESS_CLIENT_SECRET`, the same one the Kimi K3 CI reads from the
`mke-rack5-ci/mqa-dashboard` Secret) is in
`/etc/buildkite-agent/mqa-dashboard.env` (root:buildkite-agent, 0640) on each
host. A job without it fails; manual API builds can set
`MINIMAX_M31_MQA_PUSH_REQUIRED=0` to skip the push instead.

## Manual runs

Rerun the evals on an already-published image, or test CI changes from a
branch:

```bash
bk build create \
  --pipeline nho2-minimax-m31-ci \
  --branch <pr-branch> \
  --commit <full-pr-commit-sha> \
  --env MINIMAX_M31_PR_NUMBER=<pr-number> \
  --env MINIMAX_M31_REUSE_IMAGE=inferactinc/dev:<existing-tag> \
  --env MINIMAX_M31_CI_REF=<ci-branch> \
  --yes
```

`MINIMAX_M31_IMAGE_BUILD_REF` selects the `image-build` branch, and
`MINIMAX_M31_TIMEOUT_MINUTES` (default 180; a full run takes about 50 minutes) sets the eval job timeout.

The pipeline has every GitHub-event build type disabled, so it only runs
builds created through the API.
