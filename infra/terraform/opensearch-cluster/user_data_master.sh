#!/bin/bash
set -euo pipefail

# ── Master node bootstrap ─────────────────────────────────────────────────────
# Role: master only — no data, no ingest
# Instance: r7g.xlarge (4 vCPU, 32GB RAM)
# JVM heap: ${jvm_heap}
# Elasticsearch equivalent of the managed OpenSearch r7g.xlarge.search master nodes.

ES_VERSION="${elasticsearch_version}"
CLUSTER_NAME="${cluster_name}"
NODE_NAME="${node_name}"
JVM_HEAP="${jvm_heap}"
MASTER_IPS=(${master_ip_list})
MASTER_NAMES=(${master_names_list})

# ── Install Elasticsearch ─────────────────────────────────────────────────────
# ARM64 / aarch64 tarball for Graviton r7g instances

curl -fsSL "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-$${ES_VERSION}-linux-aarch64.tar.gz" \
  -o /tmp/elasticsearch.tar.gz

mkdir -p /opt/elasticsearch
tar -xzf /tmp/elasticsearch.tar.gz -C /opt/elasticsearch --strip-components=1
rm /tmp/elasticsearch.tar.gz

useradd -r -s /sbin/nologin elasticsearch || true
chown -R elasticsearch:elasticsearch /opt/elasticsearch
mkdir -p /var/lib/elasticsearch /var/log/elasticsearch
chown -R elasticsearch:elasticsearch /var/lib/elasticsearch /var/log/elasticsearch

# ── JVM heap ──────────────────────────────────────────────────────────────────

mkdir -p /opt/elasticsearch/config/jvm.options.d
cat > /opt/elasticsearch/config/jvm.options.d/heap.options <<EOF
-Xms$${JVM_HEAP}
-Xmx$${JVM_HEAP}
EOF

# ── elasticsearch.yml ─────────────────────────────────────────────────────────

PRIVATE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)

cat > /opt/elasticsearch/config/elasticsearch.yml <<EOF
cluster.name: $${CLUSTER_NAME}
node.name: $${NODE_NAME}

# Master-only node
node.roles: [master]

network.host: 0.0.0.0
http.port: 9200
transport.port: 9300

path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

# Cluster formation
discovery.seed_hosts: [$(IFS=,; echo "$${MASTER_IPS[*]}")]
cluster.initial_master_nodes: [$(IFS=,; echo "$${MASTER_NAMES[*]}")]

# Performance tuning (matching equivalent managed service settings)
indices.fielddata.cache.size: 20%
indices.query.bool.max_clause_count: 1024

# Security — disabled for VPC-internal cluster (no TLS, no auth enforcement).
# Re-enable and configure certificates if this cluster is ever exposed beyond VPC.
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

echo "Master node $${NODE_NAME} bootstrap complete at $(date)"
