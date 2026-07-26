#!/usr/bin/env bash
#
# One-time setup for qdrant memory alarms with SLACK notifications
# (run from an admin laptop).
#
# Creates: IAM permission for worker-0's role to publish CloudWatch metrics,
# an SNS topic, a small Lambda that formats CloudWatch alarm events and posts
# them to a Slack incoming webhook, and per-node alarms on the custom metrics
# published by tools/qdrant_memory_metrics.py (worker-0, 1-min timer):
#   - MemoryUsedPercent > 85 for 3 consecutive minutes
#   - SwapUsedGB > 58 for 5 consecutive minutes (near-exhaustion of the 64G
#     swapfile — the July OOM precursor; steady-state parks ~40-60G of cold
#     pages in swap, so lower thresholds flap)
#   - NodeUnreachable >= 1 for 3 consecutive minutes (frozen-box signal)
#
# Prereq: a Slack incoming webhook URL for the target channel
# (Slack: create app -> Incoming Webhooks -> Add New Webhook to Workspace).
#
# Usage:
#   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T…/B…/x… \
#     ./infra/scripts/setup-qdrant-alarms.sh
set -euo pipefail

REGION="${AWS_REGION:-us-west-1}"
WEBHOOK="${SLACK_WEBHOOK_URL:?set SLACK_WEBHOOK_URL (Slack incoming webhook)}"
NAMESPACE="CardOracle/Qdrant"
NODES=(node-0 node-1 node-2)
FN_NAME="qdrant-alerts-to-slack"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

echo "== IAM: allow worker-0's role to put metrics =="
PROFILE=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=ip-address,Values=54.67.55.75" \
  --query "Reservations[].Instances[].IamInstanceProfile.Arn" --output text | awk -F/ '{print $NF}')
ROLE=$(aws iam get-instance-profile --instance-profile-name "$PROFILE" \
  --query "InstanceProfile.Roles[0].RoleName" --output text)
echo "worker-0 role: $ROLE"
aws iam put-role-policy --role-name "$ROLE" --policy-name cloudwatch-put-metrics \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"cloudwatch:PutMetricData","Resource":"*"}]}'

echo "== SNS topic =="
TOPIC=$(aws sns create-topic --region "$REGION" --name qdrant-alerts --query TopicArn --output text)
echo "topic: $TOPIC"

echo "== Lambda execution role =="
LROLE="qdrant-alerts-slack-role"
aws iam create-role --role-name "$LROLE" --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' 2>/dev/null \
  || echo "role exists"
aws iam attach-role-policy --role-name "$LROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
echo "waiting 10s for role propagation"; sleep 10

echo "== Lambda function =="
WORKDIR=$(mktemp -d)
cat > "$WORKDIR/index.py" << 'PYEOF'
import json, os, urllib.request

EMOJI = {"ALARM": ":rotating_light:", "OK": ":white_check_mark:",
         "INSUFFICIENT_DATA": ":grey_question:"}

def handler(event, context):
    for rec in event.get("Records", []):
        msg = json.loads(rec["Sns"]["Message"])
        state = msg.get("NewStateValue", "?")
        name = msg.get("AlarmName", "?")
        reason = msg.get("NewStateReason", "")
        node = next((d["value"] for d in
                     msg.get("Trigger", {}).get("Dimensions", [])
                     if d.get("name") == "Node"), "?")
        metric = msg.get("Trigger", {}).get("MetricName", "?")
        text = (f"{EMOJI.get(state, ':bell:')} *{name}* is *{state}*\n"
                f"> node: `{node}`  metric: `{metric}`\n> {reason}")
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            os.environ["SLACK_WEBHOOK_URL"], data=body,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    return {"ok": True}
PYEOF
(cd "$WORKDIR" && zip -q fn.zip index.py)

if aws lambda get-function --region "$REGION" --function-name "$FN_NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$FN_NAME" \
    --zip-file "fileb://$WORKDIR/fn.zip" >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$FN_NAME"
  aws lambda update-function-configuration --region "$REGION" --function-name "$FN_NAME" \
    --environment "Variables={SLACK_WEBHOOK_URL=$WEBHOOK}" >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$FN_NAME"
  echo "lambda updated"
else
  aws lambda create-function --region "$REGION" --function-name "$FN_NAME" \
    --runtime python3.12 --handler index.handler \
    --role "arn:aws:iam::${ACCOUNT}:role/${LROLE}" \
    --zip-file "fileb://$WORKDIR/fn.zip" --timeout 15 \
    --environment "Variables={SLACK_WEBHOOK_URL=$WEBHOOK}" >/dev/null
  aws lambda wait function-active --region "$REGION" --function-name "$FN_NAME"
  echo "lambda created"
fi
rm -rf "$WORKDIR"

echo "== Wire SNS -> Lambda =="
aws lambda add-permission --region "$REGION" --function-name "$FN_NAME" \
  --statement-id sns-qdrant-alerts --action lambda:InvokeFunction \
  --principal sns.amazonaws.com --source-arn "$TOPIC" 2>/dev/null || echo "permission exists"
aws sns subscribe --region "$REGION" --topic-arn "$TOPIC" --protocol lambda \
  --notification-endpoint "arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FN_NAME}" >/dev/null
echo "subscribed"

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
    --alarm-description "qdrant ${node} swap >58GB for 5 min (near-full 64G swapfile)" \
    --namespace "$NAMESPACE" --metric-name SwapUsedGB \
    --dimensions "Name=Node,Value=${node}" \
    --statistic Maximum --period 60 --evaluation-periods 5 \
    --threshold 58 --comparison-operator GreaterThanThreshold \
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

echo "== Test message =="
aws lambda invoke --region "$REGION" --function-name "$FN_NAME" --payload \
  '{"Records":[{"Sns":{"Message":"{\"AlarmName\":\"qdrant-alerts-setup-test\",\"NewStateValue\":\"OK\",\"NewStateReason\":\"Setup complete — this is a test message.\",\"Trigger\":{\"MetricName\":\"Setup\",\"Dimensions\":[{\"name\":\"Node\",\"value\":\"all\"}]}}"}}]}' \
  --cli-binary-format raw-in-base64-out /dev/null >/dev/null && echo "test message sent — check the Slack channel"

echo "9 alarms created, Slack wired. Enable the metrics timer on worker-0 to start data flowing."
