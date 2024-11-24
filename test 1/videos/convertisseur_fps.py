import cv2
import os

def change_video_fps(input_path, output_dir, target_fps):
    """
    Change les FPS d'une vidéo et enregistre une nouvelle version.

    :param input_path: Chemin de la vidéo d'entrée.
    :param output_dir: Répertoire de sortie.
    :param target_fps: Fréquence d'images cible (int).
    """
    # Charger la vidéo originale
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo : {input_path}")
        return

    # Obtenir les propriétés de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Créer le chemin de sortie
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # Nom de base de la vidéo
    output_path = os.path.join(output_dir, f"{base_name}_{target_fps}FPS.mp4")

    # Initialiser l'écrivain vidéo
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    print(f"Modification de {original_fps} FPS à {target_fps} FPS pour la vidéo : {input_path}")

    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)

    cap.release()

    # Ajuster les FPS
    total_frames = len(frame_list)
    frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1

    new_frame_list = frame_list[::frame_interval]  # Échantillonnage des frames

    for frame in new_frame_list:
        out.write(frame)

    out.release()
    print(f"Vidéo enregistrée avec succès à : {output_path}")


# Fonction principale
if __name__ == "__main__":
    # Définir le chemin de la vidéo d'entrée
    current_dir = os.path.dirname(__file__)  # Répertoire actuel
    input_video = os.path.join(current_dir, "rue.mp4")  # Chemin vers la vidéo

    # Vérifier si la vidéo existe
    if not os.path.exists(input_video):
        print(f"Erreur : La vidéo '{input_video}' est introuvable.")
        exit()

    # Répertoire de sortie (même que le répertoire de la vidéo)
    output_dir = os.path.join(current_dir, "Rue")

    # Liste des FPS cibles
    target_fps_list = [15, 30, 60, 120]

    # Générer des vidéos pour chaque FPS cible
    for target_fps in target_fps_list:
        change_video_fps(input_video, output_dir, target_fps)
