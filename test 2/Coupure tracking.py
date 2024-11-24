import cv2
import os

# Fonction pour suivre et enregistrer la frame où il y a une perte de suivi ou un arrêt manuel
def save_loss_frame(video_path, tracker_name, tracker_create, output_dir, manual_roi=None):
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

    total_frames = 0
    loss_frame_path = None
    loss_frame_number = None
    reason = "Fin"  # Par défaut, la raison sera "Fin" si la vidéo se termine normalement
    last_successful_box = None
    last_valid_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # Fin de la vidéo, sauvegarder la dernière frame valide avec la boîte si elle existe
            loss_frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{tracker_name}.png")
            if last_valid_frame is not None:
                if last_successful_box is not None:
                    x, y, w, h = [int(v) for v in last_successful_box]
                    cv2.rectangle(last_valid_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(loss_frame_path, last_valid_frame)
                print(f"Vidéo terminée. Dernière frame sauvegardée : {loss_frame_path}")
            else:
                print(f"Erreur : Aucune frame valide pour '{tracker_name}' dans '{video_path}'.")
            loss_frame_number = total_frames
            break

        success, box = tracker.update(frame)
        total_frames += 1

        if success:
            # Dessiner la boîte de suivi
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            last_successful_box = box  # Enregistrer la dernière boîte valide
            last_valid_frame = frame.copy()  # Enregistrer la dernière frame valide
        else:
            # Sauvegarder la frame de perte de suivi
            loss_frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{tracker_name}.png")
            cv2.imwrite(loss_frame_path, frame)
            loss_frame_number = total_frames
            reason = "Perte"
            print(f"Perte de suivi détectée. Frame sauvegardée : {loss_frame_path}")
            break

        # Annoter la vidéo
        cv2.putText(frame, f"Tracker: {tracker_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames: {total_frames}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Afficher la vidéo
        cv2.imshow(f"Suivi - {tracker_name}", frame)

        # Quitter manuellement avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Sauvegarder la frame où l'utilisateur a arrêté le suivi
            loss_frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{tracker_name}.png")
            cv2.imwrite(loss_frame_path, frame)
            loss_frame_number = total_frames
            reason = "Manuel"
            print(f"Arrêt manuel détecté. Frame sauvegardée : {loss_frame_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    return loss_frame_path, loss_frame_number, reason

# Programme principal
if __name__ == "__main__":
    current_dir = os.getcwd()
    video_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

    if not video_files:
        print(f"Aucune vidéo trouvée dans le répertoire '{current_dir}'.")
        exit()

    # Répertoire pour sauvegarder les frames de perte de suivi
    output_dir = os.path.join(current_dir, "frames_loss_tracking")
    os.makedirs(output_dir, exist_ok=True)

    # Fichier pour enregistrer les résultats
    results_file = os.path.join(output_dir, "loss_frames.txt")

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
    with open(results_file, "w") as f:
        f.write(f"{'Tracker':<15}{'Vidéo':<25}{'Frame de Perte':<15}{'Raison':<10}\n")
        f.write("=" * 65 + "\n")

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
                loss_frame_path, loss_frame_number, reason = save_loss_frame(video_file, tracker_name, tracker_create, output_dir, roi)

                if loss_frame_number is not None:
                    f.write(f"{tracker_name:<15}{os.path.basename(video_file):<25}{loss_frame_number:<15}{reason:<10}\n")
