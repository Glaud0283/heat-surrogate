#!/usr/bin/env bash
# One-time AWS setup for the heat-surrogate pipeline. Safe to re-run.
# Usage (from the repo root, after `aws login`):
#   GITHUB_REPO=your-user/heat-surrogate infra/setup.sh
set -euo pipefail

: "${GITHUB_REPO:?Set GITHUB_REPO=owner/repo}"
REGION="${AWS_REGION:-eu-west-2}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
OIDC_ARN="arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"
ECR_REPO_ARN="arn:aws:ecr:${REGION}:${ACCOUNT}:repository/heat-serve"
echo "Account $ACCOUNT, region $REGION, repo $GITHUB_REPO"

render() {  # template file -> stdout, with placeholders filled in
  sed -e "s|__ACCOUNT__|$ACCOUNT|g" -e "s|__REGION__|$REGION|g" \
      -e "s|__OIDC_ARN__|$OIDC_ARN|g" -e "s|__ECR_REPO_ARN__|$ECR_REPO_ARN|g" \
      -e "s|__SUB__|${2:-}|g" "$1"
}
role() {  # name, trust-policy JSON (string)
  if aws iam get-role --role-name "$1" > /dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$1" --policy-document "$2"
  else
    aws iam create-role --role-name "$1" --assume-role-policy-document "$2" > /dev/null
  fi
  echo "   role $1"
}
managed() { aws iam list-policies --scope AWS --query "Policies[?PolicyName=='$1'].Arn" --output text | tr '\t' '\n' | grep -m1 '^arn:'; }

echo "1. ECR repository: immutable tags, vulnerability scan on push"
aws ecr create-repository --region "$REGION" --repository-name heat-serve \
  --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true \
  > /dev/null 2>&1 || echo "   already exists"

echo "2. Trust GitHub's OIDC token issuer"
aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com > /dev/null 2>&1 || echo "   already exists"

echo "3. Service-linked roles that ECS, load balancing and auto scaling use"
for svc in ecs.amazonaws.com elasticloadbalancing.amazonaws.com ecs.application-autoscaling.amazonaws.com; do
  aws iam create-service-linked-role --aws-service-name "$svc" > /dev/null 2>&1 || true
done

echo "4. Roles that AWS itself uses to run the service"
role heat-ecs-execution "$(render infra/trust-ecs-tasks.json)"
aws iam attach-role-policy --role-name heat-ecs-execution \
  --policy-arn "$(managed AmazonECSTaskExecutionRolePolicy)"
role heat-ecs-infrastructure "$(render infra/trust-ecs.json)"
aws iam attach-role-policy --role-name heat-ecs-infrastructure \
  --policy-arn "$(managed AmazonECSInfrastructureRoleforExpressGatewayServices)"

echo "5. Roles GitHub Actions may assume, each limited to one kind of job"
role heat-gh-push "$(render infra/trust-github.json "repo:${GITHUB_REPO}:ref:refs/heads/main")"
aws iam put-role-policy --role-name heat-gh-push --policy-name push-to-ecr \
  --policy-document "$(render infra/policy-push.json)"
for env in staging production; do
  role "heat-gh-deploy-$env" "$(render infra/trust-github.json "repo:${GITHUB_REPO}:environment:${env}")"
  aws iam put-role-policy --role-name "heat-gh-deploy-$env" --policy-name deploy-express \
    --policy-document "$(render infra/policy-deploy.json)"
done

cat <<OUT

Done. Values for GitHub (Module 06 sets them with gh):
  AWS_REGION              $REGION
  ECR_REGISTRY            ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
  AWS_PUSH_ROLE_ARN       arn:aws:iam::${ACCOUNT}:role/heat-gh-push
  ECS_EXECUTION_ROLE_ARN  arn:aws:iam::${ACCOUNT}:role/heat-ecs-execution
  ECS_INFRA_ROLE_ARN      arn:aws:iam::${ACCOUNT}:role/heat-ecs-infrastructure
  staging    AWS_DEPLOY_ROLE_ARN  arn:aws:iam::${ACCOUNT}:role/heat-gh-deploy-staging
  production AWS_DEPLOY_ROLE_ARN  arn:aws:iam::${ACCOUNT}:role/heat-gh-deploy-production
OUT
