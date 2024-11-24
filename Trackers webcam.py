import cv2

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

# Choisir le tracker
tracker_name = "CSRT"  # Remplacez par le nom du tracker souhaité
tracker = TRACKER_TYPES[tracker_name]()

# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Lire le premier cadre
ret, frame = cap.read()
if not ret:
    print("Erreur : Impossible de lire le flux vidéo.")
    exit()

# Sélectionner l'objet à suivre
roi = cv2.selectROI("Sélectionnez l'objet", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, roi)
cv2.destroyWindow("Sélectionnez l'objet")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mettre à jour le tracker
    success, box = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Echec du suivi !", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, f"Suivi avec {tracker_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.imshow("Suivi en temps réel", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
