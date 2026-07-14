import sys
sys.stdout.reconfigure(encoding='utf-8')

log_path = r"C:\Users\Jordan\.gemini\antigravity-ide\brain\4c54fda8-7612-40ef-b2eb-c3aa58003f57\.system_generated\tasks\task-1086.log"

try:
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    print("=== Task 1086 Log Content ===")
    print(content)
except Exception as e:
    print(f"Error reading log: {e}")
