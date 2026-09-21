# Upload this project to GitHub

Create an empty public repository with these values:

- Repository name: `lightweight-class-balanced-fer`
- Description: `Reproducible MobileNetV3 facial-expression recognition with class-balanced focal loss and label smoothing on FER-2013.`
- Visibility: `Public`
- Add README: `Off`
- Add .gitignore: `No .gitignore`
- Add license: `No license`

The package already contains a README, `.gitignore`, MIT license, and `CITATION.cff`.

After creating the empty repository, extract the ZIP, open a terminal inside `github_repo`, and run:

```bash
git init
git add .
git commit -m "Initial reproducible FER-2013 implementation"
git branch -M main
git remote add origin https://github.com/hindlaziri/lightweight-class-balanced-fer.git
git push -u origin main
```

If Git asks for authentication, sign in through GitHub Credential Manager or use a personal access token instead of your account password.

The `.gitignore` excludes `data_train.pt`, checkpoints, virtual environments, bytecode, and logs. Do not commit or redistribute the FER-2013 image file unless the dataset terms explicitly permit it. Before linking the repository in a journal submission, complete the matched experiments and create a tagged release such as `v1.0.0`.
