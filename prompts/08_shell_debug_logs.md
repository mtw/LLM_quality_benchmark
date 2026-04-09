Interpret the following scenario and propose a debugging sequence.

Scenario:
- A service behaves correctly when started manually in an interactive shell
- The same service behaves differently when launched by launchd or systemd
- Environment variables differ between the shell and the service context
- Model files are expected in one directory, but the service may be reading another
- The user wants to confirm whether the wrong executable, wrong user, wrong environment, or wrong model path is responsible

Output format:
1. likely root cause
2. next 5 commands
3. what result from each command would confirm or refute the hypothesis

Keep the answer concrete and operational.
