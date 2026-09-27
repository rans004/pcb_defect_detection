# ================================================================
#  PCB Defect Detection System
#  Detects and classifies manufacturing defects on Printed Circuit
#  Boards using a custom-trained YOLOv11 detection model.
#
#  Defect classes (from DeepPCB / Roboflow dataset):
#    missing_hole · mouse_bite · open_circuit
#    short · spur · spurious_copper
#
#  Requirements:
#      pip install ultralytics opencv-python numpy
#
#  Usage:
#      python pcb_defect_detector.py
#      Edit the bottom section to switch between camera / image / video
# ================================================================

import cv2
import csv
import torch
import numpy as np
from time import time, strftime
from ultralytics import YOLO


# ----------------------------------------------------------------
#  DEFECT_MAP
#  Maps each defect class → severity level, display colour (BGR),
#  and a short engineer-friendly description.
#  Update keys to match the class names in your data.yaml exactly.
# ----------------------------------------------------------------
DEFECT_MAP = {
    "missing_hole"    : {"severity": "HIGH",   "color": (0,   0,   220), "desc": "Drill hole absent"},
    "mouse_bite"      : {"severity": "MEDIUM", "color": (0,  140,  255), "desc": "Board edge erosion"},
    "open_circuit"    : {"severity": "HIGH",   "color": (50,  50,  200), "desc": "Broken conductor"},
    "short"           : {"severity": "CRITICAL","color": (0,   0,  180), "desc": "Conductor bridging"},
    "spur"            : {"severity": "LOW",    "color": (80, 200,   80), "desc": "Excess copper spur"},
    "spurious_copper" : {"severity": "MEDIUM", "color": (0,  180,  180), "desc": "Unwanted copper patch"},
}

# Severity ordering for PASS/FAIL logic
SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# Board fails inspection if any defect at or above this level found
FAIL_THRESHOLD = "MEDIUM"

# Fallback for unknown classes
DEFAULT_DEFECT = {"severity": "UNKNOWN", "color": (150, 150, 150), "desc": "Unclassified defect"}


# ----------------------------------------------------------------
#  PCBDefectDetector
# ----------------------------------------------------------------
class PCBDefectDetector:
    """
    Real-time PCB defect detection and quality inspection system.

    For each detected defect:
      - Draws a coloured bounding box with defect label and severity
      - Tracks defect counts per type in the session
      - Determines overall board PASS/FAIL status
      - Logs all detections to a timestamped CSV file
    """

    def __init__(self, source, model_path="C:\\Users\\rAns\\PCB\\runs\\detect\\PCB_detection2\\weights\\best.pt",
                 conf_threshold=0.25, log_csv=True):
        """
        Parameters
        ----------
        source         : 0 = webcam, int for other cameras,
                         str path for image or video file
        model_path     : path to trained YOLO detection model (.pt)
        conf_threshold : minimum confidence to display a detection
        log_csv        : if True, saves all detections to a CSV file
        """
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.source    = source
        self.conf      = conf_threshold
        self.log_csv   = log_csv

        print(f"[INFO] Device     : {self.device}")
        print(f"[INFO] Loading    : {model_path}")

        self.model       = YOLO(model_path)
        self.model.to(self.device)
        self.class_names = self.model.model.names
        print(f"[INFO] Classes    : {list(self.class_names.values())}")

        # Session counters — one per defect class
        self.session_counts = {name: 0 for name in self.class_names.values()}

        # CSV log setup
        if self.log_csv:
            self.csv_path = f"pcb_log_{strftime('%Y%m%d_%H%M%S')}.csv"
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "defect_type", "severity",
                    "confidence", "x1", "y1", "x2", "y2"
                ])
            print(f"[INFO] Logging to : {self.csv_path}")

    # ──────────────────────────────────────────────────────────
    #  Helper: defect info lookup
    # ──────────────────────────────────────────────────────────
    def get_defect_info(self, class_name):
        return DEFECT_MAP.get(class_name.lower().strip(), DEFAULT_DEFECT)

    # ──────────────────────────────────────────────────────────
    #  Helper: determine board status from current detections
    # ──────────────────────────────────────────────────────────
    def get_board_status(self, detected_classes):
        """Returns 'PASS', 'FAIL', or 'OK — no defects'"""
        if not detected_classes:
            return "PASS", (0, 200, 80)

        max_rank = 0
        for cls in detected_classes:
            info = self.get_defect_info(cls)
            rank = SEVERITY_RANK.get(info["severity"], 0)
            max_rank = max(max_rank, rank)

        fail_rank = SEVERITY_RANK.get(FAIL_THRESHOLD, 2)
        if max_rank >= fail_rank:
            return "FAIL", (0, 0, 220)
        return "PASS", (0, 200, 80)

    # ──────────────────────────────────────────────────────────
    #  Step 1: inference
    # ──────────────────────────────────────────────────────────
    def predict(self, frame):
        if frame.shape[2] == 4:
           frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # RGBA → BGR
        results = self.model(
            frame,
            conf=self.conf,
            verbose=False,
            device=self.device,
        )
        return results[0]

    # ──────────────────────────────────────────────────────────
    #  Step 2: draw detections on frame
    # ──────────────────────────────────────────────────────────
    def draw_detections(self, frame, result):
        """
        Draws a coloured bounding box, defect label, severity badge,
        and description for every detected defect.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return frame, []

        detected_classes = []
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(len(result.boxes)):
            class_id   = int(result.boxes.cls[i].item())
            confidence = float(result.boxes.conf[i].item())
            class_name = self.class_names[class_id]

            x1 = int(result.boxes.xyxy[i][0].item())
            y1 = int(result.boxes.xyxy[i][1].item())
            x2 = int(result.boxes.xyxy[i][2].item())
            y2 = int(result.boxes.xyxy[i][3].item())

            info     = self.get_defect_info(class_name)
            color    = info["color"]
            severity = info["severity"]
            desc     = info["desc"]

            # ── Bounding box ──────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Corner accent marks (engineering aesthetic) ───
            corner_len = 10
            for cx, cy, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 2)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 2)

            # ── Main label: "open_circuit  87%"  ─────────────
            main_label = f"{class_name}  {confidence:.0%}"
            (mw, mh), _ = cv2.getTextSize(main_label, font, 0.50, 1)

            ly = max(y1 - 26, mh + 30)
            cv2.rectangle(frame, (x1, ly - mh - 6), (x1 + mw + 6, ly + 2), color, -1)
            cv2.putText(frame, main_label, (x1 + 3, ly - 2),
                        font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Severity badge below main label ───────────────
            badge = f"[{severity}] {desc}"
            (bw, bh), _ = cv2.getTextSize(badge, font, 0.38, 1)
            by = ly + bh + 6
            cv2.rectangle(frame, (x1, by - bh - 4), (x1 + bw + 6, by + 2),
                          (30, 30, 30), -1)
            cv2.putText(frame, badge, (x1 + 3, by - 1),
                        font, 0.38, color, 1, cv2.LINE_AA)

            # ── Update session counters ───────────────────────
            self.session_counts[class_name] += 1
            detected_classes.append(class_name)

            # ── CSV logging ───────────────────────────────────
            if self.log_csv:
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        strftime("%Y-%m-%d %H:%M:%S"),
                        class_name, severity,
                        f"{confidence:.4f}",
                        x1, y1, x2, y2
                    ])

        return frame, detected_classes

    # ──────────────────────────────────────────────────────────
    #  Step 3: sidebar panel
    # ──────────────────────────────────────────────────────────
    def draw_sidebar(self, frame, detected_classes, fps):
        h       = frame.shape[0]
        panel_w = 250
        panel   = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (22, 22, 28)

        font   = cv2.FONT_HERSHEY_SIMPLEX
        white  = (210, 210, 210)
        dimmed = (100, 100, 100)
        green  = (80,  200,  80)

        # ── Header ────────────────────────────────────────────
        cv2.putText(panel, "PCB Inspector", (12, 30),
                    font, 0.58, white, 1, cv2.LINE_AA)
        cv2.putText(panel, f"FPS: {fps:.1f}", (12, 52),
                    font, 0.44, green, 1, cv2.LINE_AA)
        cv2.line(panel, (12, 64), (panel_w - 12, 64), (55, 55, 55), 1)

        # ── Board status ──────────────────────────────────────
        status, status_color = self.get_board_status(detected_classes)
        cv2.putText(panel, "Board status", (12, 84),
                    font, 0.40, dimmed, 1, cv2.LINE_AA)

        # Large status text
        (sw, sh), _ = cv2.getTextSize(status, font, 0.80, 2)
        cv2.putText(panel, status, (12, 84 + sh + 14),
                    font, 0.80, status_color, 2, cv2.LINE_AA)

        cv2.line(panel, (12, 130), (panel_w - 12, 130), (55, 55, 55), 1)

        # ── Defects found this frame ──────────────────────────
        cv2.putText(panel, "Defects (session)", (12, 150),
                    font, 0.40, dimmed, 1, cv2.LINE_AA)

        y = 172
        for class_name, count in self.session_counts.items():
            if count == 0:
                continue
            info  = self.get_defect_info(class_name)
            color = info["color"]

            cv2.circle(panel, (20, y - 3), 5, color, -1)
            cv2.putText(panel, f"{class_name}", (33, y),
                        font, 0.40, white, 1, cv2.LINE_AA)
            cv2.putText(panel, f"x{count}  [{info['severity']}]",
                        (33, y + 14), font, 0.36, color, 1, cv2.LINE_AA)
            y += 38
            if y > h - 50:
                break

        # ── Defects in current frame ───────────────────────────
        frame_count = len(detected_classes)
        cv2.line(panel, (12, h - 58), (panel_w - 12, h - 58), (55, 55, 55), 1)
        cv2.putText(panel, f"This frame: {frame_count} defect(s)",
                    (12, h - 40), font, 0.40, white, 1, cv2.LINE_AA)
        cv2.putText(panel, "ESC to quit", (12, h - 22),
                    font, 0.36, dimmed, 1, cv2.LINE_AA)

        return np.hstack([frame, panel])

    # ──────────────────────────────────────────────────────────
    #  Static image mode
    # ──────────────────────────────────────────────────────────
    def run_on_image(self, image_path, save_path="pcb_result.jpg"):
        frame = cv2.imread(image_path)
        if len(frame.shape) == 3 and frame.shape[2] == 4:
           frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame is None:
            print(f"[ERROR] Cannot read: {image_path}")
            return

        result              = self.predict(frame)
        frame, detected_cls = self.draw_detections(frame, result)
        frame               = self.draw_sidebar(frame, detected_cls, fps=0.0)

        cv2.imwrite(save_path, frame)
        print(f"[INFO] Saved → {save_path}")
        cv2.imshow("PCB Inspector", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ──────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────
    def __call__(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {self.source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        print("[INFO] Running — press ESC to quit\n")

        fps_buffer = []

        while True:
            t0 = time()
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended.")
                break
            # Fix: force to 3-channel BGR immediately after reading
            if frame.ndim == 2:
               frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
               frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) 

            result              = self.predict(frame)
            frame, detected_cls = self.draw_detections(frame, result)

            elapsed = time() - t0
            fps_buffer.append(1.0 / max(elapsed, 1e-6))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer)

            frame = self.draw_sidebar(frame, detected_cls, fps)
            cv2.imshow("PCB Inspector", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC — exiting.")
                break

        cap.release()
        cv2.destroyAllWindows()

        # ── Terminal summary ──────────────────────────────────
        print("\n" + "=" * 48)
        print("  PCB INSPECTION SUMMARY")
        print("=" * 48)
        total = sum(self.session_counts.values())
        if total == 0:
            print("  No defects detected this session.")
        else:
            for name, count in self.session_counts.items():
                if count > 0:
                    sev = self.get_defect_info(name)["severity"]
                    print(f"  {name:<22} {count:>4}x   [{sev}]")
        print(f"\n  Total defects : {total}")
        if self.log_csv:
            print(f"  Log saved     : {self.csv_path}")
        print("=" * 48)


# ================================================================
#  Entry point
# ================================================================
if __name__ == "__main__":

    detector = PCBDefectDetector(
        source="infer\\scr.png",               # 0=webcam | "pcb_video.mp4" | "image.jpg"
        model_path="C:\\Users\\rAns\\PCB\\runs\\detect\\PCB_detection2\\weights\\best.pt",   # your trained detection model
        conf_threshold=0.25,
        log_csv=True,           # saves detections to CSV
    )

    # Live camera / video mode:
    detector()

    # Static image mode (comment out the line above first):
    # detector.run_on_image("pcb_sample.jpg", save_path="pcb_result.jpg")
