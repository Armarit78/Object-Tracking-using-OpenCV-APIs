import cv2
import os

# Fonction pour enregistrer le suivi
def save_tracking_video(video_path, tracker_name, tracker_create, output_dir, manual_roi=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo '{video_path}'.")
        return None

    # Lire le premier cadre
    ret, frame = cap.read()
    if not ret:
        print(f"Erreur : Impossible de lire la vidéo '{video_path}'.")
        cap.release()
        return None

    # Initialiser le tracker avec la ROI manuelle
    roi = manual_roi
    tracker = tracker_create()
    tracker.init(frame, roi)

    # Définir le codec et créer l'écrivain vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{tracker_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Enregistrement en cours : {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, box = tracker.update(frame)
        if success:
            # Dessiner la boîte de suivi
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Annoter la vidéo
        cv2.putText(frame, f"Tracker: {tracker_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Écrire la frame dans la vidéo de sortie
        out.write(frame)

        # Afficher la vidéo en cours de traitement
        cv2.imshow(f"Suivi - {tracker_name}", frame)

        # Quitter manuellement avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Arrêt manuel détecté.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Vidéo enregistrée : {output_path}")

# Programme principal
if __name__ == "__main__":
    current_dir = os.getcwd()
    video_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

    if not video_files:
        print(f"Aucune vidéo trouvée dans le répertoire '{current_dir}'.")
        exit()

    # Répertoire pour sauvegarder les vidéos de sortie
    output_dir = os.path.join(current_dir, "videos_tracking")
    os.makedirs(output_dir, exist_ok=True)

    # Liste des trackers à tester
    TRACKER_TYPES = {
        "BOOSTING": cv2.legacy.TrackerBoosting_create,
        "MIL": cv2.legacy.TrackerMIL_create,
        "KCF": cv2.legacy.TrackerKCF_create,
        "TLD": cv2.legacy.TrackerTLD_create,
        "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
        "MOSSE": cv2.legacy.TrackerMOSSE_create,
        "CSRT": cv2.legacy.TrackerCSRT_create,
    }

    # Traiter chaque vidéo
    for video_file in video_files:
        print(f"Traitement de la vidéo : {video_file}")

        # Lire la première frame de la vidéo pour sélectionner la ROI
        cap = cv2.VideoCapture(video_file)
        ret, frame = cap.read()
        if not ret:
            print(f"Erreur : Impossible de lire la première frame de '{video_file}'.")
            cap.release()
            continue

        print("Veuillez sélectionner l'objet à suivre pour cette vidéo.")
        roi = cv2.selectROI("Sélectionner un objet", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Sélectionner un objet")
        cap.release()

        for tracker_name, tracker_create in TRACKER_TYPES.items():
            print(f"  Test du tracker : {tracker_name}")
            save_tracking_video(video_file, tracker_name, tracker_create, output_dir, roi)
