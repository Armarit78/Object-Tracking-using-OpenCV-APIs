import cv2
import psutil
import time
import os


# Fonction pour tester le tracker et collecter les métriques
def test_tracker(video_path, tracker_name, tracker_create, roi, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo '{video_path}'.")
        return None

    # Lire le premier cadre
    ret, frame = cap.read()
    if not ret:
        print(f"Erreur : Impossible de lire le flux vidéo pour '{video_path}'.")
        cap.release()
        return None

    # Initialiser le tracker avec le ROI fourni
    tracker = tracker_create()
    tracker.init(frame, roi)

    # Préparer l'écrivain vidéo pour sauvegarder la vidéo avec le suivi
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{video_name}_{tracker_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    fps_list = []
    cpu_usage = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        success, box = tracker.update(frame)
        elapsed_time = time.time() - start_time
        frame_fps = 1 / elapsed_time if elapsed_time > 0 else 0
        fps_list.append(frame_fps)

        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_usage.append(cpu_percent)

        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Echec du suivi !", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Ajouter texte sur la vidéo
        cv2.putText(frame, f"{tracker_name} FPS: {frame_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)

        # Écrire la frame dans la vidéo de sortie
        out.write(frame)

        # Afficher la vidéo
        cv2.imshow(f"Suivi - {tracker_name}", frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        total_frames += 1

    cap.release()
    out.release()  # Fermer l'écrivain vidéo
    cv2.destroyAllWindows()

    # Calculer les métriques
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

    return {
        "tracker": tracker_name,
        "avg_fps": avg_fps,
        "avg_cpu": avg_cpu,
        "total_frames": total_frames
    }


# Enregistrer les résultats dans un fichier
def save_results_to_file(results, output_file):
    with open(output_file, 'w') as file:
        # Écrire l'entête du tableau
        file.write(f"{'Tracker':<15}{'FPS Moyen':<15}{'CPU Moyen (%)':<15}{'Frames Traitées':<15}\n")
        file.write("=" * 60 + "\n")

        # Écrire chaque résultat
        for result in results:
            file.write(
                f"{result['tracker']:<15}{result['avg_fps']:<15.2f}{result['avg_cpu']:<15.2f}{result['total_frames']:<15}\n")


# Programme principal
if __name__ == "__main__":
    # Répertoire vidéo et chemin
    video_dir = os.path.join("videos", "Rue")
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

    if not video_files:
        print(f"Aucune vidéo trouvée dans le répertoire '{video_dir}'.")
        exit()

    # Répertoire de sortie pour les vidéos suivies
    output_dir = os.path.join("videos", "tracked_videos")
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

    # Résultats globaux
    all_results = []

    # Utiliser le même ROI pour toutes les vidéos
    first_roi = None

    # Tester chaque vidéo avec chaque tracker
    for video_index, video_file in enumerate(video_files):
        print(f"Traitement de la vidéo : {video_file}")

        # Demander le ROI uniquement pour la première vidéo
        if first_roi is None:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"Erreur : Impossible d'ouvrir la vidéo '{video_file}'.")
                continue

            # Lire le premier cadre
            ret, frame = cap.read()
            if not ret:
                print(f"Erreur : Impossible de lire le flux vidéo pour '{video_file}'.")
                cap.release()
                continue

            # Sélectionner l'objet dans la première vidéo
            print("Veuillez sélectionner l'objet à suivre dans la première vidéo.")
            first_roi = cv2.selectROI("Sélectionnez l'objet", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Sélectionnez l'objet")
            cap.release()

        # Tester chaque tracker avec le même ROI
        for tracker_name, tracker_create in TRACKER_TYPES.items():
            print(f"  Test du tracker : {tracker_name}")
            result = test_tracker(video_file, tracker_name, tracker_create, first_roi, output_dir)
            if result:
                all_results.append(result)

    # Enregistrer les résultats dans un fichier
    output_file = os.path.join("videos", "tracker_results.txt")
    save_results_to_file(all_results, output_file)
    print(f"Résultats enregistrés dans : {output_file}")
