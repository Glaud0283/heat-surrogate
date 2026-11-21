#!/usr/bin/env bash
# Usage: scripts/inspect.sh SERVICE [CLUSTER]
# Where to look when a deployment misbehaves: deployments, service events, container logs.
set -euo pipefail

service="$1"
cluster="${2:-default}"

echo "== Deployments (newest first)"
aws ecs describe-services --cluster "$cluster" --services "$service" \
  --query 'services[0].deployments[].[status,rolloutState,taskDefinition,runningCount,desiredCount]' \
  --output table

echo "== Last 10 service events"
aws ecs describe-services --cluster "$cluster" --services "$service" \
  --query 'services[0].events[:10].[createdAt,message]' --output text

echo "== Container logs, last 30 minutes"
group=$(aws logs describe-log-groups --log-group-name-prefix "/aws/ecs/$cluster/$service" \
          --query 'logGroups[0].logGroupName' --output text)
aws logs tail "$group" --since 30m
