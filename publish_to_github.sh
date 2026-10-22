#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# publish_to_github.sh
# One-shot publisher for the latent-phenotyping repository.
# Run from Git Bash (Windows), WSL, macOS or Linux terminal, inside the repo:
#     cd "C:/Users/manuel.cortes/Music/Prueba/DINOv2_ENTREGA_v2/DINOv2_latent_phenotyping/github_repo"
#     bash publish_to_github.sh
#
# Edit the three variables below before the first run.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GH_USER="manuelcortes17"                                          # ← CHANGE ME
REPO_NAME="dinov2-phaseolus-boeremia-latent-phenotyping"                # ← optional
INITIAL_MSG="Initial release — reproducible pipeline + manuscript"      # ← optional

# ─────────────────────────────────────────────────────────────────────────────
# 0. Preflight
# ─────────────────────────────────────────────────────────────────────────────
command -v git >/dev/null || { echo "✖ git not installed"; exit 1; }

git config --global user.name  2>/dev/null || git config --global user.name  "Manuel A. Cortés-Quiceno"
git config --global user.email 2>/dev/null || git config --global user.email "macortesq@unal.edu.co"
git config --global init.defaultBranch main

# ─────────────────────────────────────────────────────────────────────────────
# 1. Init repo & stage
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -d .git ]; then
    echo "▶ git init"
    git init -b main
fi

echo "▶ staging files…"
git add .

# Show what will be committed
echo "────────────── files to commit ──────────────"
git status --short
echo "─────────────────────────────────────────────"

# ─────────────────────────────────────────────────────────────────────────────
# 2. First commit
# ─────────────────────────────────────────────────────────────────────────────
if git log >/dev/null 2>&1; then
    echo "▶ repo already has commits — making an incremental commit"
    git commit -m "$INITIAL_MSG" || echo "nothing to commit"
else
    echo "▶ first commit"
    git commit -m "$INITIAL_MSG"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Create the remote repo on GitHub (needs `gh` CLI) OR skip if it exists
# ─────────────────────────────────────────────────────────────────────────────
if command -v gh >/dev/null; then
    if gh repo view "$GH_USER/$REPO_NAME" >/dev/null 2>&1; then
        echo "ℹ GitHub repo already exists — reusing"
    else
        echo "▶ creating GitHub repo $GH_USER/$REPO_NAME (public)"
        gh repo create "$GH_USER/$REPO_NAME" --public \
            --description "Latent-space phenotyping of Phaseolus vulgaris × Boeremia with frozen DINOv2 — code, figures and manuscript."
    fi
    git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git" 2>/dev/null || \
        git remote set-url origin "https://github.com/$GH_USER/$REPO_NAME.git"
else
    cat <<EOF
ℹ The GitHub CLI (gh) is not installed. Do one of:
  (a) install from https://cli.github.com and re-run this script, OR
  (b) create the repo manually at https://github.com/new (empty, no README),
      then run:
          git remote add origin https://github.com/$GH_USER/$REPO_NAME.git
EOF
    read -rp "Press ENTER once the empty remote repo exists and the remote is added…"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Push
# ─────────────────────────────────────────────────────────────────────────────
echo "▶ pushing to origin/main"
git push -u origin main

echo "✓ Done — https://github.com/$GH_USER/$REPO_NAME"
