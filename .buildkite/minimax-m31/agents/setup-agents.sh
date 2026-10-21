#!/bin/bash
# Installs two Buildkite agents (queue b300-tp4) on a verda B300 node, one per
# 4-GPU half. Reads "<agent-token>\n<hf-token>" on stdin. Run with sudo.
# Afterwards: `docker login -u inferactinc` as buildkite-agent (HOME is the
# shared /home/buildkite-agent, so once covers both nodes), then
# `systemctl enable --now buildkite-agent-0 buildkite-agent-1`.
set -euo pipefail

read -r AGENT_TOKEN
read -r HF_TOKEN
HOST=$(hostname -s)

if ! command -v buildkite-agent >/dev/null; then
  curl -fsSL https://keys.openpgp.org/vks/v1/by-fingerprint/32A37959C2FA5C3C99EFBC32A79206696452D198 \
    | gpg --dearmor --yes -o /usr/share/keyrings/buildkite-agent-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/buildkite-agent-archive-keyring.gpg] https://apt.buildkite.com/buildkite-agent stable main" \
    > /etc/apt/sources.list.d/buildkite-agent.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq buildkite-agent
fi
usermod -aG docker buildkite-agent
# The Docker socket's group is the directory group `eng`, which the local
# buildkite-agent user cannot join, so grant it access with an ACL.
command -v setfacl >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq acl
mkdir -p /etc/systemd/system/docker.socket.d
printf '[Socket]\nExecStartPost=/usr/bin/setfacl -m u:buildkite-agent:rw /run/docker.sock\n' \
  > /etc/systemd/system/docker.socket.d/50-buildkite-agent.conf
setfacl -m u:buildkite-agent:rw /run/docker.sock

# Builds and temp dirs live on /home (NFS); the root disk is small.
CI_HOME=/home/buildkite-agent
mkdir -p "$CI_HOME"
chown buildkite-agent:buildkite-agent "$CI_HOME"
chmod 755 "$CI_HOME"

install -d -m 0750 -o root -g buildkite-agent /etc/buildkite-agent/hooks
umask 027
cat > /etc/buildkite-agent/hooks/environment <<EOF
#!/bin/bash
export HF_TOKEN="${HF_TOKEN}"
EOF
chown root:buildkite-agent /etc/buildkite-agent/hooks/environment
chmod 0750 /etc/buildkite-agent/hooks/environment

for SLOT in 0 1; do
  if [[ $SLOT == 0 ]]; then GPUS=0,1,2,3; PORT=8000; else GPUS=4,5,6,7; PORT=8100; fi
  NAME="${HOST}-gpu${GPUS//,/}"
  GPUS_TAG="${GPUS:0:1}-${GPUS: -1}"
  mkdir -p "$CI_HOME/$NAME/builds" "$CI_HOME/$NAME/tmp"
  chown -R buildkite-agent:buildkite-agent "$CI_HOME/$NAME"
  cat > "/etc/buildkite-agent/agent-${SLOT}.cfg" <<EOF
token="${AGENT_TOKEN}"
name="${NAME}"
tags="queue=b300-tp4,host=${HOST},gpus=${GPUS_TAG}"
build-path="${CI_HOME}/${NAME}/builds"
hooks-path="/etc/buildkite-agent/hooks"
plugins-path="${CI_HOME}/${NAME}/plugins"
spawn=1
EOF
  chown root:buildkite-agent "/etc/buildkite-agent/agent-${SLOT}.cfg"
  chmod 0640 "/etc/buildkite-agent/agent-${SLOT}.cfg"

  cat > "/etc/systemd/system/buildkite-agent-${SLOT}.service" <<EOF
[Unit]
Description=Buildkite agent ${NAME} (b300-tp4, GPUs ${GPUS})
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=buildkite-agent
Environment=HOME=${CI_HOME}
Environment=TMPDIR=${CI_HOME}/${NAME}/tmp
Environment=CI_GPU_DEVICES=${GPUS}
Environment=CI_SERVE_PORT=${PORT}
Environment=CI_HF_HOME=/mnt/shared/hf-models
ExecStart=/usr/bin/buildkite-agent start --config /etc/buildkite-agent/agent-${SLOT}.cfg
RestartSec=5
Restart=on-failure
TimeoutStopSec=0
KillMode=process

[Install]
WantedBy=multi-user.target
EOF
done

# The package's default single-agent unit is not used.
systemctl disable --now buildkite-agent.service 2>/dev/null || true
systemctl daemon-reload
echo "installed agents on ${HOST}"
