# Upload this project to GitHub

Create an empty repository on GitHub, then run these commands inside this folder:

```bash
git init
git add .
git commit -m "Initial reproducible FER-2013 implementation"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

The `.gitignore` excludes `data_train.pt`, checkpoints, virtual environments, bytecode, and logs. Do not commit or redistribute the FER-2013 image file unless the dataset terms explicitly permit it. The repository contains no trained weights.

Before linking the repository in a journal submission, replace any author placeholders in the paper, complete both matched experiments, retain the generated JSON and CSV metrics required for reproducibility, and create a tagged release such as `v1.0.0`.
