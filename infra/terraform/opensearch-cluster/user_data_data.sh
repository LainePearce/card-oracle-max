#!/bin/bash
set -euo pipefail

# ── Data node bootstrap ───────────────────────────────────────────────────────
# Role: data + ingest (no master)
# Instance: c7g.2xlarge (8 vCPU, 16GB RAM)
# JVM heap: ${jvm_heap}
# EBS device: ${ebs_device} → mounted at /var/lib/elasticsearch

ES_VERSION="${elasticsearch_version}"
CLUSTER_NAME="${cluster_name}"
NODE_NAME="${node_name}"
JVM_HEAP="${jvm_heap}"
EBS_DEVICE="${ebs_device}"
MASTER_IPS=(${master_ip_list})
MASTER_NAMES=(${master_names_list})

# ── Format and mount EBS data volume ─────────────────────────────────────────

while [ ! -b "$${EBS_DEVICE}" ]; do
  echo "Waiting for EBS device $${EBS_DEVICE}..."
  sleep 3
done

if ! blkid "$${EBS_DEVICE}" | grep -q ext4; then
  mkfs.ext4 -F "$${EBS_DEVICE}"
fi

mkdir -p /var/lib/elasticsearch
mount "$${EBS_DEVICE}" /var/lib/elasticsearch
echo "$${EBS_DEVICE} /var/lib/elasticsearch ext4 defaults,nofail 0 2" >> /etc/fstab

# ── Install Elasticsearch ─────────────────────────────────────────────────────
# ARM64 / aarch64 tarball for Graviton c7g instances

curl -fsSL "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-$${ES_VERSION}-linux-aarch64.tar.gz" \
  -o /tmp/elasticsearch.tar.gz

mkdir -p /opt/elasticsearch
tar -xzf /tmp/elasticsearch.tar.gz -C /opt/elasticsearch --strip-components=1
rm /tmp/elasticsearch.tar.gz

useradd -r -s /sbin/nologin elasticsearch || true
chown -R elasticsearch:elasticsearch /opt/elasticsearch
chown -R elasticsearch:elasticsearch /var/lib/elasticsearch
mkdir -p /var/log/elasticsearch
chown -R elasticsearch:elasticsearch /var/log/elasticsearch

# ── JVM heap ──────────────────────────────────────────────────────────────────

mkdir -p /opt/elasticsearch/config/jvm.options.d
cat > /opt/elasticsearch/config/jvm.options.d/heap.options <<EOF
-Xms$${JVM_HEAP}
-Xmx$${JVM_HEAP}
EOF

# ── elasticsearch.yml ─────────────────────────────────────────────────────────

cat > /opt/elasticsearch/config/elasticsearch.yml <<EOF
cluster.name: $${CLUSTER_NAME}
node.name: $${NODE_NAME}

# Data + ingest only — master handled by dedicated nodes
node.roles: [data, ingest]

network.host: 0.0.0.0
http.port: 9200
transport.port: 9300

path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

# Point at dedicated master nodes for cluster formation
discovery.seed_hosts: [$(IFS=,; echo "$${MASTER_IPS[*]}")]
cluster.initial_master_nodes: [$(IFS=,; echo "$${MASTER_NAMES[*]}")]

# Performance tuning (matching equivalent managed service settings)
indices.fielddata.cache.size: 20%
indices.query.bool.max_clause_count: 1024

# Bulk indexing performance
indices.memory.index_buffer_size: 20%
thread_pool.write.queue_size: 1000

# Security — disabled for VPC-internal cluster.
xpack.security.enabled: false
EOF

# ── System limits ─────────────────────────────────────────────────────────────

cat >> /etc/security/limits.conf <<EOF
elasticsearch soft nofile 65536
elasticsearch hard nofile 65536
elasticsearch soft memlock unlimited
elasticsearch hard memlock unlimited
EOF

echo "vm.max_map_count=262144" >> /etc/sysctl.conf
sysctl -p

# ── Systemd service ───────────────────────────────────────────────────────────

cat > /etc/systemd/system/elasticsearch.service <<EOF
[Unit]
Description=Elasticsearch $${ES_VERSION} - $${NODE_NAME}
After=network.target

[Service]
Type=simple
User=elasticsearch
Group=elasticsearch
WorkingDirectory=/opt/elasticsearch
ExecStart=/opt/elasticsearch/bin/elasticsearch
Restart=on-failure
RestartSec=10
LimitNOFILE=65536
LimitMEMLOCK=infinity

Environment=ES_JAVA_OPTS="-Xms$${JVM_HEAP} -Xmx$${JVM_HEAP}"
Environment=ES_PATH_CONF=/opt/elasticsearch/config

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable elasticsearch
systemctl start elasticsearch

echo "Data node $${NODE_NAME} bootstrap complete at $(date)"
