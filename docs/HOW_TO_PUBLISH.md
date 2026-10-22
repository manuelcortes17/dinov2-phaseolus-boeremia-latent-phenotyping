# 🚀 How to Publish This Repository to GitHub

A step-by-step guide to upload the project to GitHub from a Windows machine with **Git Bash**.
If you prefer a one-liner, just run `bash publish_to_github.sh` from the repo root and skip this doc.

---

## 📋 Prerequisites

| Tool | Why | Install |
|---|---|---|
| **Git** | version control | <https://git-scm.com/download/win> |
| **GitHub account** | hosting | <https://github.com/signup> |
| *(optional)* **GitHub CLI** `gh` | one-command remote creation | <https://cli.github.com> |

Open **Git Bash** (right-click inside the folder → *Git Bash Here*).

---

## 1 · Configure Git once per machine

```bash
git config --global user.name  "Manuel A. Cortés-Quiceno"
git config --global user.email "macortesq@unal.edu.co"
git config --global init.defaultBranch main
```

## 2 · Create the empty GitHub repo

**Option A — web interface**

1. Go to <https://github.com/new>
2. Name: `dinov2-phaseolus-boeremia-latent-phenotyping`
3. Description: *Latent-space phenotyping of Phaseolus vulgaris × Boeremia with frozen DINOv2 — code, figures and manuscript.*
4. **Public** · *do NOT* initialize with README, LICENSE or .gitignore (we already have them).
5. Click **Create repository**.

**Option B — GitHub CLI**

```bash
gh auth login
gh repo create manuelcortes17/dinov2-phaseolus-boeremia-latent-phenotyping --public \
  --description "Latent-space phenotyping of Phaseolus vulgaris × Boeremia with frozen DINOv2."
```

## 3 · Initialize the local repo

Open Git Bash inside the project folder:

```bash
cd "/c/Users/manuel.cortes/Music/Prueba/DINOv2_ENTREGA_v2/DINOv2_latent_phenotyping/github_repo"
git init -b main
git add .
git status                      # check what will be committed
git commit -m "Initial release — reproducible pipeline + manuscript"
```

## 4 · Connect to the remote and push

```bash
git remote add origin https://github.com/manuelcortes17/dinov2-phaseolus-boeremia-latent-phenotyping.git
git branch -M main
git push -u origin main
```

> If GitHub asks for a password, use a **Personal Access Token** instead
> (<https://github.com/settings/tokens> → fine-grained → repo scope).

## 5 · Verify

Open `https://github.com/manuelcortes17/dinov2-phaseolus-boeremia-latent-phenotyping` — the README is rendered at the bottom with the banner, badges and figure grid.

---

## 🔁 Day-to-day workflow

```bash
# after you edit a figure, caption or script
git add -A
git commit -m "polish fig 8C quadrant labels"
git push
```

## 🏷 Tagging a frozen release (recommended before submission)

```bash
git tag -a v1.0 -m "Submission to Journal of Experimental Botany"
git push origin v1.0
```

Then on GitHub → *Releases* → *Draft new release* → pick `v1.0` → attach the final `.docx` if you want.

## 🌐 Optional: enable GitHub Pages for the Quarto notebook

```bash
# render the html locally
quarto render analysis_notebook.qmd
# commit the single-file html
git add analysis_notebook.html
git commit -m "add rendered notebook"
git push
```

Then on GitHub → *Settings* → *Pages* → **Deploy from branch** → `main` / `/root`. The notebook will be served at `https://manuelcortes17.github.io/dinov2-phaseolus-boeremia-latent-phenotyping/analysis_notebook.html`.

---

## 🧯 Troubleshooting

| Problem | Fix |
|---|---|
| `remote: Repository not found` | wrong URL — check username & repo name |
| `error: failed to push some refs` | run `git pull --rebase origin main` first, then push again |
| `fatal: refusing to merge unrelated histories` | you initialized the GitHub repo with a README — easiest fix is to delete the remote repo and recreate it empty |
| Push stalls on big files | run `git lfs install` and track `*.docx` / `*.png` with Git LFS — or just keep the current layout (none of our files exceed GitHub's 100 MB limit) |
| Need to update manuscript DOCX | overwrite `docs/manuscript/*.docx`, then `git add`, `git commit`, `git push` |

---

Done — your repository is live.
