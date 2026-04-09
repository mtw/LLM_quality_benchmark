Interpret the following situation and propose the next 5 shell commands to run.

Situation:
- On the main machine, `ollama list` shows many installed models
- From another machine, an SSH-forwarded connection to port 11434 appears to show only a subset of models
- The user suspects that a different Ollama instance or a different models directory is being queried remotely
- The environment may differ between an interactive shell and a service launched by launchd or systemd

Rules:
- For each command, explain in one sentence what it checks
- Prefer commands that distinguish process, port, environment, and model path issues
- Assume a Unix-like shell
- Be specific
- Do not repeat equivalent commands
