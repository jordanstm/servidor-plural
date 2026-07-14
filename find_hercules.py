import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

folder = r"C:\ProjetoCSharp"
search_term = "hercules"

print("=== Searching for hercules in C:\\ProjetoCSharp ===")
count = 0
for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith((".cs", ".xaml", ".csproj", ".sln", ".json", ".xml", ".config")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if search_term in content.lower():
                    print(f"Found in {path}")
                    count += 1
                    if count >= 30:
                        print("Too many results, stopping search.")
                        sys.exit(0)
            except Exception as e:
                pass
