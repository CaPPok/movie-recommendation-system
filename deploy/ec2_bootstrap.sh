#!/usr/bin/env bash
#
# Prepare a fresh EC2 instance to run the retraining job on a schedule.
#
#   sudo bash deploy/ec2_bootstrap.sh https://github.com/<user>/<web-repo>.git
#
# Tested against Amazon Linux 2023 and Ubuntu 22.04 on t3.large or larger.
#
# Sizing: the ALS fit holds a 262k x 17.6k sparse matrix plus two float32 factor
# matrices, which peaks around 6 GB. t3.large (8 GB) is the smallest instance
# that finishes; t3.xlarge is the comfortable choice. Storage needs ~20 GB for
# the dataset and a few artifact versions.
#
# Credentials: attach an IAM **instance role**, do not copy access keys onto the
# box. boto3 picks the role up automatically. The role needs s3:GetObject,
# s3:PutObject and s3:ListBucket on the project bucket, plus dynamodb:Scan on
# the Interactions table if this instance also runs the export.

set -euo pipefail

REPO_URL="${1:-}"
APP_USER="${APP_USER:-ec2-user}"
APP_HOME="${APP_HOME:-/opt/movie-rec}"
# The ML project is a subdirectory of the web repository; see
# docs/aws_deployment.md section 6.
PROJECT_SUBDIR="${PROJECT_SUBDIR:-movie-recommendation-system}"
BRANCH="${BRANCH:-main}"

if [[ -z "${REPO_URL}" ]]; then
  echo "Usage: sudo bash deploy/ec2_bootstrap.sh <git-repo-url>" >&2
  exit 2
fi

echo "==> Cài gói hệ thống"
if command -v dnf >/dev/null 2>&1; then
  dnf install -y git python3.11 python3.11-pip gcc gcc-c++ make
  PYTHON=python3.11
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y git python3.11 python3.11-venv python3-pip build-essential
  PYTHON=python3.11
else
  echo "Không nhận diện được package manager (cần dnf hoặc apt-get)." >&2
  exit 1
fi

id -u "${APP_USER}" >/dev/null 2>&1 || useradd --create-home "${APP_USER}"

echo "==> Lấy mã nguồn"
mkdir -p "${APP_HOME}"
if [[ -d "${APP_HOME}/repo/.git" ]]; then
  git -C "${APP_HOME}/repo" fetch --depth 1 origin "${BRANCH}"
  git -C "${APP_HOME}/repo" reset --hard "origin/${BRANCH}"
else
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${APP_HOME}/repo"
fi

PROJECT_DIR="${APP_HOME}/repo/${PROJECT_SUBDIR}"
if [[ ! -f "${PROJECT_DIR}/retrain.py" ]]; then
  echo "Không thấy ${PROJECT_DIR}/retrain.py. Đặt PROJECT_SUBDIR cho đúng." >&2
  exit 1
fi

echo "==> Tạo virtualenv"
"${PYTHON}" -m venv "${APP_HOME}/venv"
"${APP_HOME}/venv/bin/pip" install --upgrade pip
"${APP_HOME}/venv/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"
"${APP_HOME}/venv/bin/pip" install -r "${PROJECT_DIR}/requirements-aws.txt"

chown -R "${APP_USER}:${APP_USER}" "${APP_HOME}"

echo "==> Cài systemd timer"
install -m 0644 "${PROJECT_DIR}/deploy/movie-rec-retrain.service" \
  /etc/systemd/system/movie-rec-retrain.service
install -m 0644 "${PROJECT_DIR}/deploy/movie-rec-retrain.timer" \
  /etc/systemd/system/movie-rec-retrain.timer

# The unit files carry placeholders so they stay readable in the repository.
sed -i "s#@APP_USER@#${APP_USER}#g;s#@APP_HOME@#${APP_HOME}#g;s#@PROJECT_DIR@#${PROJECT_DIR}#g" \
  /etc/systemd/system/movie-rec-retrain.service

systemctl daemon-reload
systemctl enable --now movie-rec-retrain.timer

cat <<EOF

==> Xong.

Kiểm tra:
  systemctl list-timers movie-rec-retrain.timer
  sudo -u ${APP_USER} ${APP_HOME}/venv/bin/python ${PROJECT_DIR}/retrain.py --dry-run

Chạy ngay một lần, không đợi lịch:
  systemctl start movie-rec-retrain.service
  journalctl -u movie-rec-retrain.service -f

Cấu hình region/bucket (nếu khác configs/aws.yaml):
  /etc/systemd/system/movie-rec-retrain.service  ->  Environment=
EOF
