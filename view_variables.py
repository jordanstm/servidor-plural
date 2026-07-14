import sys
sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\ProjetoCSharp\Prado.Licitacao\ConversorECOWinthor\MainWindow.xaml.cs"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(90 - 1, 140):
    print(f"{i+1}: {lines[i]}", end="")
