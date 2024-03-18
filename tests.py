import subprocess

def run_command(command):
    try:
        # Exécute la commande, mais redirige la sortie standard et la sortie d'erreur vers le néant
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, shell=True)
        print(f"\033[1;32mLe lancement de la commande '{command}' a abouti .")
    except subprocess.CalledProcessError as e:
        print(f"\033[1;31Erreur lors de l'exécution de la commande '{command}': {e}")

# Liste des commandes à exécuter
commands_to_run = [
    "python3 random_projection.py",
    "python3 test_ecoindex.py http://www.google.fr",
    "python ComputeRMSE.py",
]

# Exécute chaque commande dans la liste
for command in commands_to_run:
    run_command(command)
