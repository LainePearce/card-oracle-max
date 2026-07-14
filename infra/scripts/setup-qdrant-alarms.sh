#!/usr/bin/env bash
#
# One-time setup for qdrant memory alarms (run from an admin laptop).
#
# Creates: IAM permission for worker-0's role to publish CloudWatch metrics,
# an SNS topic + email subscription, and per-node alarms on the custom metrics
# published by tools/qdrant_memory_metrics.py (worker-0, 1-min timer):
#   - MemoryUsedPercent > 85 for 3 consecutive minutes
#   - SwapUsedGB > 40 for 5 consecutive minutes (active thrash signal)
#   - NodeUnreachable >= 1 for 3 consecutive minutes (frozen-box signal)
#
# Usage:
#   ALERT_EMAIL=130point@gmail.com ./infra/scripts/setup-qdrant-alarms.sh
set -euo pipefail

REGION="${AWS_REGION:-us-west-1}"
EMAIL="${ALERT_EMAIL:?set ALERT_EMAIL}"
NAMESPACE="CardOracle/Qdrant"
NODES=(node-0 node-1 node-2)

echo "== IAM: allow worker-0's role to put metrics =="
PROFILE=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=ip-address,Values=54.67.55.75" \
  --query "Reservations[].Instances[].IamInstanceProfile.Arn" --output text | awk -F/ '{print $NF}')
ROLE=$(aws iam get-instance-profile --instance-profile-name "$PROFILE" \
  --query "InstanceProfile.Roles[0].RoleName" --output text)
echo "worker-0 role: $ROLE"
aws iam put-role-policy --role-name "$ROLE" --policy-name cloudwatch-put-metrics \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"cloudwatch:PutMetricData","Resource":"*"}]}'

echo "== SNS topic + email subscription =="
TOPIC=$(aws sns create-topic --region "$REGION" --name qdrant-alerts --query TopicArn --output text)
aws sns subscribe --region "$REGION" --topic-arn "$TOPIC" --protocol email --notification-endpoint "$EMAIL" >/dev/null
echo "topic: $TOPIC  (CONFIRM THE SUBSCRIPTION EMAIL sent to $EMAIL)"

echo "== Alarms =="
for node in "${NODES[@]}"; do
  aws cloudwatch put-metric-alarm --region "$REGION" \
    --alarm-name "qdrant-${node}-memory-high" \
    --alarm-description "qdrant ${node} memory >85% for 3 min" \
    --namespace "$NAMESPACE" --metric-name MemoryUsedPercent \
    --dimensions "Name=Node,Value=${node}" \
    --statistic Maximum --period 60 --evaluation-periods 3 \
    --threshold 85 --comparison-operator GreaterThanThreshold \
    --treat-missing-data notBreaching \
    --alarm-actions "$TOPIC" --ok-actions "$TOPIC"

  aws cloudwatch put-metric-alarm --region "$REGION" \
    --alarm-name "qdrant-${node}-swap-thrash" \
    --alarm-description "qdrant ${node} swap >40GB for 5 min" \
    --namespace "$NAMESPACE" --metric-name SwapUsedGB \
    --dimensions "Name=Node,Value=${node}" \
    --statistic Maximum --period 60 --evaluation-periods 5 \
    --threshold 40 --comparison-operator GreaterThanThreshold \
    --treat-missing-data notBreaching \
    --alarm-actions "$TOPIC" --ok-actions "$TOPIC"

  aws cloudwatch put-metric-alarm --region "$REGION" \
    --alarm-name "qdrant-${node}-unreachable" \
    --alarm-description "qdrant ${node} SSH-unreachable for 3 min (frozen-box signal)" \
    --namespace "$NAMESPACE" --metric-name NodeUnreachable \
    --dimensions "Name=Node,Value=${node}" \
    --statistic Maximum --period 60 --evaluation-periods 3 \
    --threshold 1 --comparison-operator GreaterThanOrEqualToThreshold \
    --treat-missing-data breaching \
    --alarm-actions "$TOPIC" --ok-actions "$TOPIC"
done
echo "9 alarms created. Done — enable the metrics timer on worker-0 to start data flowing."
