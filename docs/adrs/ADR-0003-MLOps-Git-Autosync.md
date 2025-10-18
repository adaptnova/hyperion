# ADR-0003: Implement Git Version Control and Autosync
Status: Accepted
Context: Rapid iteration on agent scripts without version control led to loss of state and difficulty tracking changes. Manual synchronization is error-prone. Need for auditable history and automated backup.
Decision:
1.  Consolidate all project code, configuration, documentation, and artifact pointers into a single `/data/hyperion` directory.
2.  Initialize a Git repository in `/data/hyperion`.
3.  Create a private GitHub repository (`adaptnova/hyperion`).
4.  Use `gh cli` for authentication and interaction with GitHub.
5.  Implement an `autosync.sh` script using `git add .`, `git commit`, and `git push origin main`.
6.  Run `autosync.sh` via a background loop (`autosync_loop.sh` launched with `nohup`) instead of `cron` due to permission issues.
7.  Use `.gitignore` to exclude large artifacts (checkpoints, logs, cloned repos, secrets).
Consequences:
* (+) Provides a versioned history of all changes.
* (+) Enables collaboration and rollback.
* (+) Automates backup of critical code and configuration to GitHub.
* (-) Background loop is less robust than a system service or correctly configured cron job (potential future improvement).
